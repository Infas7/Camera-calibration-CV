import glob
import numpy as np
import cv2
import math
from tqdm import *

def load_camera_params(file_path):
    cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    CM1 = cv_file.getNode("cameraMatrix1 - Left camera intrinsics").mat()
    CM2 = cv_file.getNode("cameraMatrix2 - Right camera intrinsics").mat()
    dist1 = cv_file.getNode("distCoeffs1 - Left camera distortion coefficients").mat()
    dist2 = cv_file.getNode("distCoeffs2 - Right camera distortion coefficients").mat()
    R = cv_file.getNode("Rotation Matrix").mat()
    T = cv_file.getNode("Translation Vector").mat()
    E = cv_file.getNode("Essential Matrix").mat()
    F = cv_file.getNode("Fundamental Matrix").mat()
    cv_file.release()
    return CM1, CM2, dist1, dist2, R, T, E, F

def load_images(images_path):
    images_files = glob.glob(images_path)
    images = list()
    for image_file in images_files:
        img = cv2.imread(image_file)
        img = cv2.normalize(img, None, alpha=1.0, beta=200, norm_type=cv2.NORM_MINMAX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)
    return images


def remap(l_map1, l_map2, r_map1, r_map2, calibrated_images_left, calibrated_images_right):
    rectified_left = list() 
    rectified_right = list()
    
    for left_img, right_img in zip(calibrated_images_left, calibrated_images_right):
        left_image = cv2.remap(left_img, l_map1, l_map2, cv2.INTER_LINEAR)
        right_image = cv2.remap(right_img, r_map1, r_map2, cv2.INTER_LINEAR)

        left_image = cv2.resize(left_image, (612,512), interpolation = cv2.INTER_AREA)
        right_image = cv2.resize(right_image, (612,512), interpolation = cv2.INTER_AREA)

        rectified_left.append(left_image) 
        rectified_right.append(right_image)
    return rectified_left, rectified_right

def SAD_algo(y, x, block_left, right_array, block_size=5):
    search_block_size = 56
    x_min = max(0, x - search_block_size)
    x_max = min(right_array.shape[1], x + search_block_size)

    min_sad = float('inf')
    min_index = None

    for x_current in range(x_min, x_max):
        block_right = right_array[y: y + block_size, x_current: x_current + block_size]

        sad = np.sum(abs(block_left - block_right)) if block_left.shape == block_right.shape else float('inf')

        if sad < min_sad:
            min_sad = sad
            min_index = (y, x_current)
    return min_index


def NCC_algo(y, x, block_left, right_array, block_size=5):
    search_block_size = 56
    x_min = max(0, x - search_block_size)
    x_max = min(right_array.shape[1], x + search_block_size)
    first_itr = True
    max_ncc = None
    max_index = None
   
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size, x: x+block_size]
        if block_left.shape != block_right.shape:
            ncc = -1
        else:
            ncc = (np.sum(block_left*block_right)/ math.sqrt(np.sum(pow(block_left, 2))* np.sum(pow(block_right, 2))))
        
        if first_itr:
            max_ncc = ncc
            max_index = (y, x)
            first_itr = False
        elif ncc > max_ncc:
            max_ncc = ncc
            max_index = (y, x)
    return max_index

def calculate_disparity_map_NCC_or_SAD(left_array, right_array):
    width =612
    height = 512
    block_size = 100
    left_array = np.array(left_array)
    right_array = np.array(right_array)
    disparity_map = np.zeros((width, height))

    for y in tqdm(range(block_size, width- block_size)):
        for x in range(block_size, height- block_size):
            block_left = left_array[y:y + block_size, x:x + block_size]
            
            # min_index = NCC_algo(y, x, block_left, right_array, block_size=block_size)
            min_index = SAD_algo(y, x, block_left, right_array, block_size=block_size)
            
            disparity_map[y, x] = abs(min_index[1] - x)
    
    # cv2.imwrite("disparity_NCC.png", disparity_map)
    cv2.imwrite("disparity_SAD.png", disparity_map)


def calculate_disparity_map_SGBM(left_image, right_image):
    n = 1
    window_size = 5

    for left_img, righ_img in zip(left_image, right_image):
        stereo = cv2.StereoSGBM_create(
            minDisparity=-25,
            numDisparities=160,
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=25,
            uniquenessRatio=30,
            speckleWindowSize=0,
            speckleRange=0,
            preFilterCap=4,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity_map = stereo.compute(left_img,righ_img).astype(np.float32)/16.0
        cv2.imwrite("SGBM_" + str(n) + ".png",disparity_map)
        n = n + 1


if __name__ == '__main__':
    CM1, CM2, dist1, dist2, R, T, E, F = load_camera_params('Parameters/cameraparameters.txt')
    left_images = load_images("SceneImages/left*.bmp")
    right_images = load_images("SceneImages/right*.bmp")
    
    rectification_L, rectification_R, projection_L, projection_R, dispartityToDepthMap, leftROI, rightROI = cv2.stereoRectify(CM1, dist1, CM2, dist2, (2448,2048), R, T,
                                                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0)
    map1x, map1y = cv2.initUndistortRectifyMap(CM1, dist1, rectification_L, projection_L, (2448,2048), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(CM2, dist2, rectification_R, projection_R, (2448,2048), cv2.CV_16SC2)
    
    rectified_l, rectified_r = remap(map1x, map1y, map2x, map2y, left_images, right_images)

    # calculate_disparity_map_NCC_or_SAD(rectified_l, rectified_r)
    calculate_disparity_map_SGBM(rectified_l, rectified_r)
