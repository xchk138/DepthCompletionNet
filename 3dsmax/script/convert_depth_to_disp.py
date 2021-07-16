import numpy as np
import cv2
import glob
import os
import platform

def depth2disp(depth_u8:np.ndarray, d_min, d_max, fx, baseline, max_disp):
    assert depth_u8.dtype == np.uint8
    assert len(depth_u8.shape)==3
    assert depth_u8.shape[2]==4
    assert d_max > d_min
    # refine the depth for sky from alpha channel
    mask_alpha = depth_u8[:,:,3]
    depth_32f = np.array(depth_u8[:,:,0], np.float32)
    depth_32f[mask_alpha==0] = 0
    d_min = max(d_min, 1e-6)
    depth_32f = (255.0 - depth_32f)/255.0*(d_max - d_min) + d_min
    disp_32f = (baseline*fx)/depth_32f
    disp_32f = np.minimum(disp_32f, max_disp)
    return np.uint8((disp_32f * 255.0 )/max_disp)


if __name__ == '__main__':
    root_path = 'E:/Gits/Datasets/DCN/scene01'
    if platform.system()=="Windows":
        root_dir_win32 = root_path.replace('/', '\\\\')
        os.system('md ' + root_dir_win32 + '\\\\disp')
        #print('md ' + root_dir_win32 + '\\\\disp')
    else:
        os.system('mkdir -p ' + root_path + '/disp')
    files = glob.glob(root_path + '/depth/*.png')

    for i in range(len(files)):
        im_depth = cv2.imread(files[i], -1)
        im_disp = depth2disp(im_depth, 0, 30000, 355.096, 10.0, 16)
        if platform.system()=="Windows":
            cv2.imwrite(files[i].replace('depth\\', 'disp\\'), im_disp)
        else:
            cv2.imwrite(files[i].replace('depth/', 'disp/'), im_disp)