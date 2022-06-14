# Stereo configuration files for Ekofisk stereo cameras.

Example code to read the camera matrices in the .xml files:

  import cv2 # Import OpenCV
  
  im = cv2.imread('LEFT_cam_51.tif', 0) # Read undistorted image
  
  kp = 'intrinsics_00.xml'
  cv_file = cv2.FileStorage(kp, cv2.FILE_STORAGE_READ)
  K = cv_file.getNode("intrinsics_penne").mat() # Intrinsic matrix
  kd = 'distortion_00.xml'
  cv_file = cv2.FileStorage(kd, cv2.FILE_STORAGE_READ)
  dist = cv_file.getNode("intrinsics_penne").mat() # Distortion vector
