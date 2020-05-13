# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:36:24 2020

@author: john
"""

import cv2
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
  fn = glob(image_path+'splitter/bottom_back_*.png')
  N = len(fn)
  print(N)
  
  des_height = 960
  des_length = 1920
  for image_number in range(1,N+1):
    
      fn1 = glob(image_path+'EquiRectProj/eqrect_bottom_%04d' %image_number+'*.png')
      fn2 = glob(image_path+'EquiRectProj/eqrect_top_%04d' %image_number+'*.png')
      
      
      # Reading raw images
      print(fn1[0],fn2[0])
      bottom_img = cv2.imread(fn1[0])  
      top_img = cv2.imread(fn2[0])  
      timestamp = (fn1[0].rsplit('_',1))[1]
      print(timestamp)


      # Calculation of depth
      sinvnfs = np.zeros((des_height,des_length), dtype=np.float32)
      for u in range(des_length):
          sinvnfs[:,u] = np.sin(np.arange(des_height, dtype=np.float32) * np.pi / des_height) 
           
     
      imgR = rotate_ccw_90(bottom_img)
      imgL = rotate_ccw_90(top_img)
      

      ''' computing hole free disparity using wls-filter      '''
      # SGBM Parameters -----------------
      window_size = 5                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
      stereo = cv2.StereoSGBM_create(
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
      
      disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
      disp = rotate_ccw_90(disp)

      valid = (disp>0)

      
      depth = 0.398 * sinvnfs / np.sin(disp*np.pi/des_height)
      depth = depth * valid + 0 * (1 - valid)
      '''distance = np.clip(distance, 0, 10)
      
      #Saving in mm.
      '''
      MIN_DEPTH = 1
      MAX_DEPTH = 100.0
      depth = depth * 1000
      depth = np.where(depth < MIN_DEPTH*1000, 0.0, depth)
      depth = np.where(depth == np.inf, 0.0, depth)
      depth = np.where(depth == np.nan, 0.0, depth)
      depth = np.where(depth > MAX_DEPTH*1000, MAX_DEPTH*1000, depth)
     
      
            
      cv2.imwrite(image_path+'/depth_SGBM/depth_%04d_' %image_number+ str(timestamp), depth.astype(np.uint16))
      cv2.imwrite(image_path+'/disparity_SGBM/disparity_sgbm_%04d_' %image_number+ str(timestamp) , disp.astype(np.uint16))

      
      # Displaying depth image
      #depth = cv2.normalize(src=depth, dst=depth, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
      #depth = np.uint8(depth)
      #cv2.imshow('Depth Image', depth)
      cv2.imwrite(image_path+'/depth_SGBM/depthNormalized_sgbm.png' , depth)

       
      # Displaying depth image
      #disp = cv2.normalize(src=disp, dst=disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
      #disp = np.uint8(disp)
      #cv2.imshow('Disparity Image', disp)
      cv2.imwrite(image_path+'/disparity_SGBM/disparityNormalized_sgbm.png' , disp)

      #cv2.waitKey()
      #cv2.destroyAllWindows()
     