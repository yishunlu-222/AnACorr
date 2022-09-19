import cv2
import os
import pdb

# file = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/16010_tomobar_cropped/multi_label/16010_tomopy_multilabel_184.tiff'
# img = cv2.imread(file)
# img2 = cv2.flip(img,0)
# cv2.imshow('img',img)
# cv2.imshow('img2',img2)
# cv2.waitKey(0)
# pdb.set_trace()
path = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_astra_cropped_8bit/multi_label'
path_save = 'D:/lys/studystudy/phd/absorption_correction/dataset/0_all_labels/14116_astra_cropped_8bit/multi_label_rotate90_anti'
try:
    os.mkdir(path_save)
except:
    pass

for i in os.listdir(path):

    if 'tiff' in i:
        img = cv2.imread(os.path.join(path, i))

        img_rotate = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)  #cv2.ROTATE_90_COUNTERCLOCKWISE cv2.ROTATE_180

        cv2.imwrite(os.path.join(path_save, i), img_rotate)
    print(i)