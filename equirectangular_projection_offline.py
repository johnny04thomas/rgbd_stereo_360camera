# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:22:05 2020

@author: john
"""
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from glob import glob
import yaml

global image_path
image_path = "data/circle/"

des_height = 960
des_length = 1920

def read_ucm_params_kalibr(filename, camid='cam0'):
    with open(filename, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    dp = data[camid]['distortion_coeffs']
    D = np.array(dp)
    pp = data[camid]['intrinsics']
    K = np.eye(3)
    K[0][0] = pp[1]
    K[1][1] = pp[2]
    K[0][2] = pp[3]
    K[1][2] = pp[4]
    xi = np.array(pp[:1])
    if (camid == 'cam0'):
        return xi, K, D
    else:
        T = data[camid]['T_cn_cnm1']
        return xi, K, D, T
    
def equirectangular_projection(img, map1, map2):
    undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return undistorted_img
        
def toSphereRad(theta, phi):
    out = np.zeros((3, des_length),dtype=float)
    out[0,:] = np.sin(theta) * np.cos(phi)
    out[1,:] = np.sin(phi)
    out[2,:] = np.cos(theta) * np.cos(phi)
    return out
    
def create_spherical_proj(K, xi, D, plus_theta, zi, rho_limit, R, R1):
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    mapx = np.zeros((des_height, des_length), dtype=np.float32)
    mapy = np.zeros((des_height, des_length), dtype=np.float32)
    height, width = mapx.shape
    step_theta = 2*np.pi / width
    step_phi = np.pi / height
    for i in range(height):
        j = np.array(range(width))
        theta = j * step_theta - np.pi + plus_theta
        phi = i * step_phi - np.pi/2
        d = toSphereRad(theta, phi)
        rho = np.arccos(d[2,:])
        d = R1.dot(d)
        d[2,:] += zi
        d = (R).dot(d)
        imagePoints, _ = cv2.omnidir.projectPoints(np.reshape(np.transpose(d), (1,des_length,3)), rvec, tvec, K, xi, D)
        ip = np.transpose(np.reshape(imagePoints, (des_length, 2)))
        ip[:,rho>rho_limit] = -1
        mapx[i,:] = ip[0,:]
        mapy[i,:] = ip[1,:]
    mask = mapx != -1
    mapx, mapy = cv2.convertMaps(mapx, mapy, cv2.CV_16SC2)
    return mapx, mapy, mask

def simpleBlend(front, back):
  
    global f_mask
    global b_mask
    global intersect
    global not_intersect

    s = front + back
    hs = front/2 + back/2
    r2 = cv2.bitwise_and(hs, hs, mask = intersect)
    r1 = cv2.bitwise_and(s, s, mask = not_intersect)
    result = r1 + r2
    return result

def rotate_cw_90(img):
    img_t = cv2.transpose(img)
    img_f = cv2.flip(img_t, flipCode=1)
    return img_f

def rotate_ccw_90(img):
    img_t = cv2.transpose(img)
    img_f = cv2.flip(img_t, flipCode=0)
    return img_f

def skew(x):
    if (isinstance(x,np.ndarray) and len(x.shape)>=2):
        return np.array([[0, -x[2][0], x[1][0]],
                         [x[2][0], 0, -x[0][0]],
                         [-x[1][0], x[0][0], 0]])
    else:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
                         
def rectify_param(T_front, T_back):
      
    n = np.array([[0], [-1], [0]])
    R_tf_bf = np.array(T_front)[:3,:3]
    t_tf_bf = np.array(T_front)[:3,3]
    t_tf_bf = t_tf_bf/np.linalg.norm(t_tf_bf)
    omega_tf_bf = np.arccos(n.transpose().dot(t_tf_bf))
    a_tf_bf = np.cross(n,t_tf_bf, axis=0)
    K_tf_bf = skew(a_tf_bf)
    Rv_tf_bf = np.eye(3) + np.sin(omega_tf_bf)*K_tf_bf + (1-np.cos(omega_tf_bf))*K_tf_bf.dot(K_tf_bf)
    Rbr_front = Rv_tf_bf.dot(R_tf_bf)
    Rtr_front = Rv_tf_bf
      
    R_tb_bb = np.array(T_back)[:3,:3]
    t_tb_bb = np.array(T_back)[:3,3]
    t_tb_bb = t_tb_bb/np.linalg.norm(t_tb_bb)
    omega_tb_bb = np.arccos(n.transpose().dot(t_tb_bb))
    a_tb_bb = np.cross(n,t_tb_bb, axis=0)
    K_tb_bb = skew(a_tb_bb)
    Rv_tb_bb = np.eye(3) + np.sin(omega_tb_bb)*K_tb_bb + (1-np.cos(omega_tb_bb))*K_tb_bb.dot(K_tb_bb)
    Rbr_back = Rv_tb_bb.dot(R_tb_bb)
    Rtr_back = Rv_tb_bb
    return Rbr_front, Rtr_front, Rbr_back, Rtr_back

if __name__ == '__main__':
  fn = glob(image_path+'splitter/bottom_back_*.png')
  N = len(fn)
  N = 64
  ''' Reading calibration parameters '''
  rho_limit= 95

  camchain_front = 'config/camchain-2020-03-31-13-06-26_front.yaml'
  camchain_back = 'config/camchain-2020-03-31-12-08-15_back.yaml'
  camchain = 'config/camchain-stereo_2019-07-05-08-24-01.yaml'
  
  bottom_fcid = 'cam0'
  Bxi_f, BK_f, BD_f = read_ucm_params_kalibr(camchain_front, bottom_fcid)
  rho_limit = np.pi/2 * rho_limit/90.0
  bottom_bcid = 'cam0'
  Bxi_b, BK_b, BD_b = read_ucm_params_kalibr(camchain_back, bottom_bcid)
  
  top_fcid = 'cam1'
  Txi_f, TK_f, TD_f, T_tf_bf = read_ucm_params_kalibr(camchain_front, top_fcid)
  top_bcid = 'cam1'
  Txi_b, TK_b, TD_b, T_tb_bb = read_ucm_params_kalibr(camchain_back, top_bcid)
  
  # getting extrinsic paramters between fisheye lenses in a single camera
  with open(camchain, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
  Ttop_b_f = data['cam1']['T_cn_cnm1']
  Tbottom_b_f = data['cam3']['T_cn_cnm1']
  ttop_b_f = np.array(Ttop_b_f)[:3,3]
  
  tbottom_b_f = np.array(Tbottom_b_f)[:3,3]
  baseline_bottom = np.linalg.norm(tbottom_b_f) 
  baseline_top = np.linalg.norm(ttop_b_f)
  Rbr_front, Rtr_front, Rbr_back, Rtr_back = rectify_param(T_tf_bf, T_tb_bb)
  
  
  for image_number in range(N,N+1):
      #fn = glob(image_path+"bottom_back_%04d" %image_number )
    
      file_img_bb = glob(image_path+'splitter/bottom_back_%04d' %image_number+'*.png')
      file_img_bf = glob(image_path+'splitter/bottom_front_%04d' %image_number+'*.png')
      file_img_tb = glob(image_path+'splitter/top_back_%04d' %image_number+'*.png')
      file_img_tf = glob(image_path+'splitter/top_front_%04d' %image_number+'*.png')
    
      # Reading raw images
      # print(fn1[0],fn2[0],fn3[0],fn4[0])
      img_bb = cv2.imread(file_img_bb[0])  
      img_bf = cv2.imread(file_img_bf[0])  
      img_tb = cv2.imread(file_img_tb[0])
      img_tf = cv2.imread(file_img_tf[0])  
      timestamp = (file_img_bb[0].rsplit('_',1))[1]
      print(timestamp)
      
      
      Bmap1_f, Bmap2_f, Bf_mask = create_spherical_proj(BK_f, Bxi_f, BD_f, 0, 0, rho_limit, np.linalg.inv(Rbr_front), np.eye(3))
      Bmap1_b, Bmap2_b, Bb_mask = create_spherical_proj(BK_b, Bxi_b, BD_b, np.pi, -baseline_bottom, rho_limit, np.linalg.inv(Rbr_back),np.eye(3)) #Rbottom_b_f)
     
      intersect = np.array(Bf_mask * Bb_mask, dtype=np.uint8)
      not_intersect = 1 - intersect
      
      bottom_front_eqimg = equirectangular_projection(img_bf,  Bmap1_f, Bmap2_f)
      bottom_back_eqimg  = equirectangular_projection(img_bb,  Bmap1_b, Bmap2_b)
   
  
      Tmap1_f, Tmap2_f, Tf_mask = create_spherical_proj(TK_f, Txi_f, TD_f, 0, 0, rho_limit, np.linalg.inv(Rtr_front), np.eye(3))
      Tmap1_b, Tmap2_b, Tb_mask = create_spherical_proj(TK_b, Txi_b, TD_b, np.pi, -baseline_top, rho_limit, np.linalg.inv(Rtr_back), np.eye(3)) #Rtop_b_f)
      
      top_front_eqimg = equirectangular_projection(img_tf, Tmap1_f, Tmap2_f)
      top_back_eqimg  = equirectangular_projection(img_tb, Tmap1_b, Tmap2_b)
  
      bottom_both_eqimg = simpleBlend(bottom_front_eqimg, bottom_back_eqimg)

      cv2.imwrite(image_path+'/EquiRectProj/eqrect_bottom_%04d_' %image_number+ str(timestamp) , bottom_both_eqimg )


      top_both_eqimg = simpleBlend(top_front_eqimg, top_back_eqimg)
      
      cv2.imwrite(image_path+'/EquiRectProj/eqrect_top_%04d_' %image_number + str(timestamp) , top_both_eqimg )
      rect_img = np.concatenate([top_both_eqimg, bottom_both_eqimg], axis=0) 
      cv2.imwrite(image_path+'RectifiedImages/rectified_image_%04d.png' %image_number, rect_img)
     
      cv2.namedWindow('Rectified equiprojections',cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Rectified equiprojections', des_height,des_length/2)
      cv2.imshow('Rectified equiprojections', rect_img)
      cv2.waitKey(0)
