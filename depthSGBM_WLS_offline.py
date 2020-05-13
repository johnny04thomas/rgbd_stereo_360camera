# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:36:24 2020

@author: john
"""

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from glob import glob
#from sklearn.preprocessing import normalize

global image_path
image_path = "data/circle/"


def rotate_cw_90(img):
    img_t = cv2.transpose(img)
    img_f = cv2.flip(img_t, flipCode=1)
    return img_f

def rotate_ccw_90(img):
    img_t = cv2.transpose(img)
    img_f = cv2.flip(img_t, flipCode=0)
    return img_f

if __name__ == '__main__':
  fn = glob(image_path+'EquiRectProj/eqrect_bottom_*.png')
  N = len(fn)
  print(N)

  for image_number in range(N,N+1):
    
      fn1 = glob(image_path+'EquiRectProj/eqrect_bottom_%04d' %image_number+'*.png')
      fn2 = glob(image_path+'EquiRectProj/eqrect_top_%04d' %image_number+'*.png')
      
      # Reading raw images
      print(fn1[0],fn2[0])
      bottom_img = cv2.imread(fn1[0])  
      top_img = cv2.imread(fn2[0])  
      timestamp = (fn1[0].rsplit('_',1))[1]
      print(timestamp)

      # Calculation of depth
      des_height = 960
      des_length = 1920
      sinvnfs = np.zeros((des_height,des_length), dtype=np.float32)
      for u in range(des_length):
          sinvnfs[:,u] = np.sin(np.arange(des_height, dtype=np.float32) * np.pi / des_height) 
           
      
      imgR = rotate_ccw_90(bottom_img)
      imgL = rotate_ccw_90(top_img)
      

      ''' computing hole free disparity using wls-filter      '''
      # SGBM Parameters -----------------
      window_size = 5                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
      left_matcher = cv2.StereoSGBM_create(
          minDisparity=0,
          numDisparities=112,             # max_disp has to be dividable by 16 f. E. HH 192, 256
          blockSize=3,
          P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
          P2=32 * 3 * window_size ** 2,
          disp12MaxDiff=2,
          uniquenessRatio=15,
          speckleWindowSize=100,
          speckleRange=2,
          preFilterCap=63,
      )
      
      right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
      # FILTER Parameters
      lmbda = 8000
      sigma = 1.2
      visual_multiplier = 1.0
      
      wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
      wls_filter.setLambda(lmbda)
      wls_filter.setSigmaColor(sigma)
      
      print('computing disparity...')
      displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/ 16.0  # .astype(np.float32)/16
      dispr = right_matcher.compute(imgR, imgL) #.astype(np.float32)/16.0
      displ = np.int16(displ)
      dispr = np.int16(dispr)
      filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
     
      filteredImg = np.float32(filteredImg)/16.0
      ''' disparity to depth'''
      filteredImg = rotate_cw_90(filteredImg)

      valid = (filteredImg>0)
   

  
      depth = 0.398 * sinvnfs / np.sin(filteredImg*np.pi/des_height)
      depth = depth * valid + 0 * (1 - valid)
      '''distance = np.clip(distance, 0, 10)
      
      #Saving in mm.
      '''
      MIN_DEPTH = 1
      MAX_DEPTH = 20.0
      depth = depth * 1000
      depth = np.where(depth < MIN_DEPTH*1000, 0.0, depth)
      depth = np.where(depth == np.inf, 0.0, depth)
      depth = np.where(depth == np.nan, 0.0, depth)
      depth = np.where(depth > MAX_DEPTH*1000, MAX_DEPTH*1000, depth)
   
      cv2.imwrite(image_path+'/depth_SGBMWLS/depthWLS_%04d_' %image_number+ str(timestamp), depth.astype(np.uint16))
      cv2.imwrite(image_path+'/disparity_SGBMWLS/disparity_sgbmWLS_%04d_' %image_number+ str(timestamp) , filteredImg.astype(np.uint16))
      
      
      # Displaying depth image
      depth = cv2.normalize(src=depth, dst=depth, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
      depth = np.uint8(depth)
      cv2.imshow('Depth Image', depth)
      cv2.imwrite(image_path+'/depth_SGBMWLS/depthN_sgbmWLS.png' , depth)

       
      # Displaying depth image
      filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
      filteredImg = np.uint8(filteredImg)
      cv2.imshow('Disparity Image', filteredImg)
      cv2.imwrite(image_path+'disparity_SGBMWLS/disparityN_sgbmWLS.png' , filteredImg)

      #cv2.waitKey()
      #cv2.destroyAllWindows()
     
     
