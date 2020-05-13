# rgbd_stereo_360camera
Data can be collected from spherical cameras using package - https://github.com/RobInLabUJI/ricoh_camera 

Sample data set can be downloaded from : https://mega.nz/folder/4jRhSCJT#11EQ6TAumTvJ5U2cRDIeDg

## Quick Use
1> equirectanguar_projection_offline.py  -  run this script to obtain equirectangular projection from raw fisheye images from all four lenses. Make sure to use the latest calibration file, preferably stored in config folder.

2> depthSGBM_offline.py - run this script to obtain depth image from stereo equirectangular image pair. Basic SGBM algorithm is used for estimating disparity.

3> depthSGBM_WLS_offline.py -  run this script to obtain better output for depth image as it uses a WLS filter in addition to default SGBM algorithm.
