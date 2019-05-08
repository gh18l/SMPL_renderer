import smpl_np
import core
import numpy as np
import cv2

def render_naked(theta, beta, tran):
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    render_result = renderer(verts, cam=None, img=None, do_alpha=False)  ## alpha channel
    return render_result

def render_naked_imgbg(theta, beta, tran, img, camera):
    """
    camera = [focal, cx, cy, trans]
    """
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    camera_for_render = np.hstack([camera[0], camera[1], camera[2], camera[3]])
    render_result = renderer(verts, cam=camera_for_render, img=img, do_alpha=False)
    return render_result

def render_naked_rotation(theta, beta, tran, angle):
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    render_result = renderer.rotated(verts, angle, cam=None, img_size=None)
    return render_result

def main():
    theta = np.zeros(72)
    theta[0] = np.pi
    beta = np.ones(10) * .03
    tran = np.zeros(3)
    render_result = render_naked(theta, beta, tran)
    cv2.imshow("view", render_result)
    cv2.waitKey()

if __name__ == '__main__':
    main()