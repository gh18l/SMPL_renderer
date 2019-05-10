This is a repository that render model and texture with SMPL model.
## Installation method
Anyone can use Anaconda(https://www.anaconda.com/) to create a environment:
```
conda create -n SMPL_renderer python=2.7
source activate SMPL_renderer
```
Then the dependecies must be installed:
```
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pip==8.1.1
pip install opendr==0.76
pip install --upgrade pip
```
## Demo
Run the demo with the 0-pose and a random shape:
```
python SMPL_renderer.py
``` 
## TODO
+ To render texture using OpenGL