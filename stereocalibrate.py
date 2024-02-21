import cv2
import numpy as np
import glob

rows = 6 
columns = 9 

def calibrate(images_folder):
    read_images = glob.glob(images_folder)
    out_images = []
    images = []
    img_points = [] # 2d points
    obj_points = [] # 3d points

    for img in read_images:
        image = cv2.imread(img, 1)
        images.append(image)

    objp = np.zeros((rows* columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
    objp = 22.1 * objp #using world scale
 
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    for i, frame in enumerate(read_images):
        image = cv2.imread(frame, 1)
        result, img_points, obj_points = draw_corners(image, obj_points, img_points, objp)
        out_images.append(result)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)
    return mtx, dist, img_points, obj_points, out_images


def draw_corners(image, obj_points, img_points, objp):
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (rows, columns), None)
    if ret:
        corners = cv2.cornerSubPix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(image, (rows,columns), corners, ret)
        obj_points.append(objp)
        img_points.append(corners)
        return image, img_points, obj_points


def draw_lines(left_map_X, left_map_Y, right_map_X, right_map_Y, imgs_L, imgs_R):
    n = 1
    for o_left, o_right in zip(imgs_L, imgs_R):  
        good_pt_left = cv2.remap(o_left, left_map_X, left_map_Y, cv2.INTER_LINEAR)
        good_pt_right = cv2.remap(o_right, right_map_X, right_map_Y, cv2.INTER_LINEAR)

        concat_image = cv2.hconcat([good_pt_left, good_pt_right])

        cropped_image = concat_image[300:1800, 600:4300]
        height = cropped_image.shape[0]
        width = cropped_image.shape[1]
        half_w = int(width / 2)

        for i in range(0, height, 10):
            cv2.line(cropped_image, (0, i), (width, i), (0, 255, 0))
            cv2.line(cropped_image, (half_w, 0), (half_w, height), (0, 255, 0))

        cv2.imwrite("RectifiedImages/rectified_" + str(n) + ".png", cropped_image)
        n = n+1


def generate_outputs(output_text_file, CM1, dist1, CM2, dist2, R, T, E, F):
    file = cv2.FileStorage(output_text_file, cv2.FILE_STORAGE_WRITE)
    file.write("cameraMatrix1 - Left camera intrinsics", CM1)
    file.write("distCoeffs1 - Left camera distortion coefficients", dist1)
    file.write("cameraMatrix2 - Right camera intrinsics", CM2)
    file.write("distCoeffs2 - Right camera distortion coefficients", dist2)
    file.write("Rotation Matrix", R)
    file.write("Translation Vector", T)
    file.write("Essential Matrix", E)
    file.write("Fundamental Matrix", F)
    file.release()

if __name__ ==  "__main__":
    mtx_L, dist_L, left_image_points, obj_points_1, out_images_L = calibrate(images_folder = 'Chessboard/left*.bmp')
    mtx_R, dist_R, right_image_points, obj_points_2, out_images_R = calibrate(images_folder = 'Chessboard/right*.bmp')

    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(obj_points_1, left_image_points, right_image_points, mtx_L, dist_L,
                                                                    mtx_R, dist_R, (rows, columns), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), 
                                                                    flags = cv2.CALIB_FIX_INTRINSIC)

    generate_outputs('Parameters/cameraparameters.txt', CM1, dist1, CM2, dist2, R, T, E, F)
    rect_L, rect_R, prjc_L, prj_R, dispartity2depth_map, leftROI, rightROI = cv2.stereoRectify(CM1, dist1, CM2, dist2, (2448,2048), R, T, 
                                                                                               np.zeros(shape=(3, 3)), np.zeros(shape=(3, 3)),
                                                                                               np.zeros(shape=(3, 3)), np.zeros(shape=(3, 3)), 
                                                                                               Q=None,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0.88)
    map1_x, map1_y = cv2.initUndistortRectifyMap(CM1, dist1, rect_L, prjc_L, (2448,2048), cv2.CV_16SC2)
    map2_x, map2_y = cv2.initUndistortRectifyMap(CM2, dist2, rect_R, prj_R, (2448,2048), cv2.CV_16SC2)
    draw_lines(map1_x, map1_y, map2_x, map2_y, out_images_L, out_images_R)
    print("Completed...")