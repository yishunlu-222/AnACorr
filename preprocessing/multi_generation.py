import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
np.set_printoptions(threshold = np.inf)
import os
import pdb
import sys
import re
from matplotlib import pyplot as plt
from meta import train_lists_group,load_lists
from utils import mask2rgb
import skimage.io as io
import tqdm
# cite by https://blog.csdn.net/qq_41895190/article/details/82791426
# set red thresh
#red
lower_red=np.array([0,43,46])
upper_red=np.array([10,255,255])
#green
lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])
#blue
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
#cyan
lower_cyan = np.array([78, 43, 46])
upper_cyan = np.array([99, 255, 255])
#liquor, the background hsv is the lower
lower_liquor = np.array([89,30,203])
upper_liquor = np.array([91,32,205])

lower_yellow = np.array([26,43,46])
upper_yellow = np.array([34,255,255])
#13284 lower_loop=np.array([90,103,159]) upper_loop=np.array([92,105,161])
# parser = argparse.ArgumentParser()
# parser.add_argument('--baseroot', '-br', type=str,
#                     default='D:/lys/studystudy/phd/absorption correction/dataset/Python3DFirst/blender/laser_slicer-master/testing/batch_ssh0/',
#                     help='the root for executing functin')
# args = parser.parse_args()

def find_color(m_root,which_pic):
    for img_list in os.listdir(m_root):
        if which_pic in img_list:
            path = os.path.join(m_root, img_list)
            img = cv2.imread(path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            fig = plt.figure()
            plt.imshow(hsv)
            plt.show()
            cv2.imwrite('test.tiff',img)
            pdb.set_trace()

def label_col_change(txt_whole,buuu=False,
                     col_range=[[lower_red,upper_red,],[lower_liquor,upper_liquor],[lower_green,upper_green],[lower_yellow,upper_yellow]],
                     names=['crystal_mask','liquor_mask','loop_mask','bu_mask']):
    """
    :param m_root:
    :param col_range:
    :param names:
    :return: new folder with standard segmentation mask
    """
    img_list =load_lists(txt_whole)
    for u,group in enumerate(img_list):

        root=group[1]
        name =os.path.basename(root).replace('crystal_mask','multilabel')
        baseroot = os.path.join(os.path.dirname(os.path.dirname(root)),'multi_label')
        try:
            os.mkdir(baseroot)
        except:
            pass
        cr=np.expand_dims(io.imread(group[1]),0)
        lo=np.expand_dims(io.imread(group[2]),0)
        li=np.expand_dims(io.imread(group[3]),0)

        if buuu is True:
            bu=np.expand_dims(io.imread(group[4]),0)
            multi=np.concatenate((li,lo,cr,bu),axis=0)
        else:
            multi = np.concatenate((li, lo, cr), axis=0)
        # pdb.set_trace()
        for i in range(len(multi)):
            mas = multi[i]
            mas = np.where(mas != 0, i + 1, mas)
            multi[i] = mas

        multi = np.amax(multi, axis=0, )
        mulit_l = mask2rgb(multi)

        filename = name
        filepath = os.path.join(baseroot, filename)
        print('{} is saved'.format(filename))

        cv2.imwrite(filepath, mulit_l)



def mask_genertator(o_root,m_root,col_range=[[lower_red,upper_red,],[lower_liquor,upper_liquor],[lower_green,upper_green],[lower_yellow,upper_yellow]],
                    names=['crystal_mask','liquor_mask','loop_mask','bu_mask']):
    # generate mask
    # col_range =
    for i,name in enumerate(names):
        low=col_range[i][0]
        upper=col_range[i][1]
        baseroot= os.path.join(os.path.dirname(o_root),name)
        try:
            os.mkdir(baseroot)
            # so it will be like /root/crystal_mask
        except:
            pass
        for img_list in os.listdir(m_root):
            path = os.path.join(m_root,img_list)
            img = cv2.imread(path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get mask
            mask = cv2.inRange(hsv, low, upper)
            if  'liquor' in name:
                mask = 255- mask
            img_list = img_list.replace('.tiff','')
            index=img_list.split('_')[-1]
            img_list = img_list.split('_')
            del img_list[-1]
            prefix = '_'.join(img_list)
            # pdb.set_trace()
            filename= '{}_{}_{}.tiff'.format(prefix,name,index)
            filepath=os.path.join(baseroot,filename)
            print(filepath)
            cv2.imwrite(filepath, mask)


def change_names(root,prefix='13295_tomobar'):
    for img_list in os.listdir(root):
        if 'tif' in img_list:
            old =  os.path.join(root,img_list)
            # img_list=os.path.splitext(img_list)[0]
            # abc = img_list.split('_')[-1][1:]

            new= os.path.join(root,'{}_{}.tiff'.format(prefix,int(re.findall(r'_\d+', os.path.basename(img_list))[-1][1:])))

            os.rename(old,new)
            print(new)


if __name__ == '__main__':
    # blue is loop , red is crystal, liquor requires to read the backgrou   nd and then 255-
    #  the origin_root, and the mask_root, and the prefix name of the crystal, the colour of the mask respectively '
    #  are need to be changed

    # all the roots are basename
    # o_root->
    #       -> img.tiff
    #       -> img2.tiff
    o_root = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_astra_cropped_v2/u8_14116_astra_cropped900'
    m_root = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16010_segmentation_labels/tiffs'
    m_root = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_segmentation_tiffs/segmentation_tiffs_full_size'

    # change_names(o_root,'13668_tomobar_whole')
    # change_names( m_root , '14116_v2_tomopy' )
    # find_color(m_root,which_pic='505')


    # pdb.set_trace()
    # #[np.array([90,103,159]),np.array([92,105,161])] [28,124,236]  [87,182,224] [0,255,255]
    # mask_genertator(o_root,m_root,[[np.array([148,108,232]),np.array([150,110,234])],[lower_liquor,upper_liquor],[np.array([85,124,138]),np.array([87,126,140])],
    #                                [np.array([25,179,242]),np.array([27,181,244])]],
    #                 names=['crystal_mask','liquor_mask','loop_mask','bubble_mask'])
    # #
    root = os.path.dirname(o_root)
    txt_train= train_lists_group(root, root, root,bu=True,multi=False)

    label_col_change(txt_train,buuu=True)
    # dir1 = '/home/yishun/dataset/segment/13295_tomobar_cropped/loop_mask/'
