import numpy as np
import pickle


class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      v_template = params['v_template']
      #v_template = remove_template_handfoot(v_template, params['weights'])
      self.v_template = v_template
      self.v_template_RT = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def get_nohandfoot_faces(self):
    with open("./smpl/models/bodyparts.pkl", 'rb') as f:
      v_ids = pickle.load(f)
    hands = np.concatenate((v_ids['hand_r'], v_ids['hand_l']))
    return np.array(filter(lambda face: np.intersect1d(face, hands).size == 0, f))

  def set_params(self, pose=None, beta=None, trans=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Prameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped_RT = self.shapedirs.dot(self.beta) + self.v_template_RT
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped_RT)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    # remove the transformation due to the rest pose
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))

    # b = []
    # a = np.array([[1.0,0.0,0.], [0.,1.,0.], [0.,0.,1.]])
    # for i in range(6890):
    #   if abs(T[i,2,3]-0.0) > 0.005:
    #     b.append(T[i,:,:])
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).tiny)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_to_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  def output_verts(self):
    return self.verts

  def get_verts(self, pose, beta, trans):
    pose = pose.reshape((24, 3))
    beta = beta.reshape(10, )
    trans = trans.reshape(3, )
    self.set_params(beta=beta, pose=pose, trans=trans)
    return self.output_verts()


  def get_nonrigid_smpl_template(self, verts, pose, beta, trans):
    v_remove_trans = verts - trans.reshape([1, 3])
    pose = pose.squeeze()
    beta = beta.squeeze()

    v_shaped = self.shapedirs.dot(beta) + self.v_template
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = pose.reshape((-1, 1, 3))
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0] - 1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
          )
        )
      )
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
      )
    )
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    verts_t = np.copy(v_remove_trans)
    for i in range(len(T)):
      T_rotation = np.linalg.inv(T[i, :3, :3])
      T_translation = np.matmul(-T_rotation, T[i, :3, 3])[:, np.newaxis]
      T_new = np.concatenate([T_rotation, T_translation], axis=1)
      rest_shape_h = np.hstack((verts_t[i, :], np.ones(1)))
      #T_translation = T[i, :3, 3].T
      #v = v_remove_trans[i, :] - T_translation
      v = np.matmul(T_new, rest_shape_h)
      verts_t[i, :] = v.T
    # for i in range(len(T)):
    #   rest_shape_h = np.hstack((verts_t[i, :], np.ones(1)))
    #   #T_translation = T[i, :3, 3].T
    #   #v = v_remove_trans[i, :] - T_translation
    #   v = np.matmul(T[i, :, :], rest_shape_h)[:3]
    #   verts_t[i, :] = v.T
    v_shaped = verts_t - self.posedirs.dot(lrotmin)
    verts_t = v_shaped - self.shapedirs.dot(beta)
    return verts_t

  def get_template(self):
    return self.v_template

  def set_template(self, template):
    self.v_template = template

def remove_template_handfoot(_template, weights):
  template = np.copy(_template)
  lefthands_index = [20, 22]
  righthands_index = [21, 23]
  lefttoes_index = [10]
  righttoes_index = [11]
  body_parsing_idx = []  ###body head
  _lefthands_idx = np.zeros(6890)
  _righthands_idx = np.zeros(6890)
  _lefttoes_idx = np.zeros(6890)
  _righttoes_idx = np.zeros(6890)
  placeholder_idx = np.zeros(6890)
  _test_idx = np.zeros(6890)
  for _, iii in enumerate(lefthands_index):
      length = len(weights[:, iii])
      for ii in range(length):
          if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _lefthands_idx[ii] = 1
            placeholder_idx[ii] = 1
  lefthands_idx = np.where(_lefthands_idx == 1)
  body_parsing_idx.append(lefthands_idx)

  for _, iii in enumerate(righthands_index):
      length = len(weights[:, iii])
      for ii in range(length):
          if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _righthands_idx[ii] = 1
            placeholder_idx[ii] = 1
  righthands_idx = np.where(_righthands_idx == 1)
  body_parsing_idx.append(righthands_idx)

  for _, iii in enumerate(lefttoes_index):
      length = len(weights[:, iii])
      for ii in range(length):
          if weights[ii, iii] > 0.3 and placeholder_idx[ii] == 0:
            _lefttoes_idx[ii] = 1
            placeholder_idx[ii] = 1
  lefttoes_idx = np.where(_lefttoes_idx == 1)
  body_parsing_idx.append(lefttoes_idx)

  for _, iii in enumerate(righttoes_index):
      length = len(weights[:, iii])
      for ii in range(length):
          if weights[ii, iii] > 0.4 and placeholder_idx[ii] == 0:
            _righttoes_idx[ii] = 1
            placeholder_idx[ii] = 1
  righttoes_idx = np.where(_righttoes_idx == 1)
  body_parsing_idx.append(righttoes_idx)

  ##lefthand
  a = body_parsing_idx[0]
  min = 99999999
  indexxx = 0
  z_mm = []
  for i in range(len(a[0])):
    index = a[0][i]
    if template[index, 0] > 0.7 and template[index, 0] < 0.72:
      z_mm.append(template[index, 2])
  z_mm = np.array(z_mm).squeeze()
  z_min = np.min(z_mm)
  z_max = np.max(z_mm)
  for i in range(len(a[0])):
    index = a[0][i]
    z = template[index, 2]
    if z > z_max:
      template[index, 2] = z_max
    if z < z_min:
      template[index, 2] = z_min
  for i in range(len(a[0])):
    index = a[0][i]
    x = template[index, 0]
    if x < min:
      min = x
      indexxx = index
  for i in range(len(a[0])):
    index = a[0][i]
    template[index, 0] = min

  ##righthand
  max = -99999999
  a = body_parsing_idx[1]
  for i in range(len(a[0])):
    index = a[0][i]
    z = template[index, 2]
    if z > z_max:
      template[index, 2] = z_max
    if z < z_min:
      template[index, 2] = z_min
  for i in range(len(a[0])):
    index = a[0][i]
    x = template[index, 0]
    if x > max:
      max = x
      indexxx = index
  for i in range(len(a[0])):
    index = a[0][i]
    template[index, 0] = max

  ### lefttoes
  min = 99999999
  a = body_parsing_idx[2]
  for i in range(len(a[0])):
    index = a[0][i]
    z = template[index, 2]
    if z < min:
      min = z
      indexxx = index
  for i in range(len(a[0])):
    index = a[0][i]
    template[index, 2] = min

  ### righttoes
  min = 99999999
  a = body_parsing_idx[3]
  for i in range(len(a[0])):
    index = a[0][i]
    z = template[index, 2]
    if z < min:
      min = z
      indexxx = index
  for i in range(len(a[0])):
    index = a[0][i]
    template[index, 2] = min
  return template

