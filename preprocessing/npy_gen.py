"""
projection data method is inspired by
https://stackoverflow.com/questions/26739670/plotting-the-projection-of-3d-plot-in-three-planes-using-contours
and tutorial:
https://www.youtube.com/watch?v=5jQVQE6yfio

the algorithm is to 1. reconstruct the 3D labelling( RGBK) 2. find a plane to get imaginary paths to do ac
3.according to the individual imaginary paths to calculate the corresponding sum of the absorption rate of all
individual paths 4. times back the sum of the absorption rate to correct voxel value
"""
from pylab import *
import skimage.io as io

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb
import cv2
import numpy as np
import skimage.transform as trans
# generate some sample data

# mg = io.imread('D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16000_tomobar_cropped/multi_label/16000_tomopy_multilabel_330.tiff')
# io.imshow(mg)
# plt.show()
# mg = mg[200:800,280:880]
# io.imshow(mg)
# plt.show()

def projection(object,angle):
    # based on the skimage.transform.rotate(2D) and it rotates about y axis
    # r_matrix=np.array([[cos(angle),0,sin(angle)],
    #                    [0, 1, 0],
    #                    [-sin(angle),0,cos(angle)]])
    if object.dtype == 'uint8':
        object=object.astype(np.float64)/255
    Z,Y,X=object.shape
    for i in range(Y):
        object[:,i,:]=trans.rotate(object[:,i,:],angle,mode='constant', cval=0,)

    proj =np.max(object,axis=0)

    return proj

def rgb2mask(rgb, COLOR=None):
    """

    :param bgr: input mask to be converted to rgb image
    :param COLOR: 1:liquor,blue; 2: loop, green ; 3: crystal, red
    :return: rgb image
    """
    if COLOR is None:
        COLOR = {0: [0, 0, 0], 1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0],4: [255, 255, 0]}
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k, v in COLOR.items():
        mask[np.all(rgb == v, axis=2)] = k

    return mask

def mask2rgb(mask, COLOR=None):
    """

    :param mask: input mask to be converted to rgb image
    :param COLOR: 1:liquor,blue; 2: loop, green ; 3: crystal, red
    :return: rgb image
    """
    if COLOR is None:
        COLOR = {0: [0, 0, 0], 1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0],4: [255, 255, 0]}

    rgb=np.zeros(mask.shape+(3,),dtype=np.uint8)

    for i in np.unique(mask):
        rgb[mask==i]=COLOR[i]

    return rgb

def save_npy(path,reverse,filename='13304_label_1C.npy',label=True,crop=False):
    """

    :param path: path should directed to image path
    :param filename:
    :param label:
    :param crop:  #[y1:y2,x1:x2]
    :return:
    """
    na = []
    for root,dir,files in os.walk(path):
        for file in files:
            if 'tif' in file:
                na.append(os.path.join(root,file))

    def take_num(ele):
        return  int(ele.split('.')[0].split('_')[-1])
    # sort the list according to the last index of the filename
    na.sort(key=take_num,reverse=reverse)
    # pdb.set_trace()

    for i,file in enumerate(na):

        if i ==0:
            file = os.path.join(path,file)
            img = io.imread(file)

            if crop:
                img = img[crop[0]:crop[1],crop[2]:crop[3]] #[y1:y2,x1:x2]
            if label:
                img=rgb2mask(img)
                img = img.astype(np.int8)
            img = np.expand_dims(img, axis=0)
            stack=img
            # pdb.set_trace()
        else:

            # index = file.split('.')[0][-4:].lstrip('0')
            index =  file.split('.')[0].split('_')[-1]

            # assert i == int(index)
            file = os.path.join(path,file)
            img = io.imread(file)

            if crop:
                img = img[crop[0]:crop[1],crop[2]:crop[3]] #[y1:y2,x1:x2]
            if label:
                img = rgb2mask(img)

                img = img.astype(np.int8)
            img = np.expand_dims(img, axis=0)
            stack = np.concatenate((stack,img),axis=0)
            print('{} is attached'.format(index))
    if label:
        stack_int = stack.astype(np.int8)
        np.save(filename,stack_int)
    else:
        np.save(filename, stack)


path='./'
path_13304 ='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/13304_tomobar_u8/multi_label_f_croppedloop'
path_16000='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16000_tomobar_cropped/multi_label_f'
path_16010 ='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16010_tomobar_cropped/multi_label_f_croppedloop'
path_14116 ='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_astra_cropped_8bit/multi_label_f'
path_14116_v2 ='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_astra_cropped_v2/multi_label'
path_13668 ='D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/13668_u8_tomobar/multi_label'
filename='{}_tomobar_cropped_v2_r.npy'.format('14116')
save_npy(path_14116_v2 ,filename=filename,label=True,reverse=True)

img_list = np.load(os.path.join(path,filename))

slice = img_list[1000,:,:]
pdb.set_trace()
# slice = mask2rgb(np.round(slice))
plt.imshow(slice)
plt.axis('off')
plt.show()
pdb.set_trace()
# image=img_list[500,:,:,:]
# pdb.set_trace()
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imshow('image window', image)
# cv2.waitKey(0)




