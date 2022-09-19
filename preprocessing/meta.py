
import numpy as np
import os
import pdb


def load_lists(txtpath,mode='train'):
    lists=[]
    txt = open(txtpath,'r')
    if mode =='eval':
        for line in txt.readlines():
            line = line.strip('\n')
            lists.append(line)
    else:
        for line in txt.readlines():
            line = line.strip('\n')
            line = line.split('  ')
            if len(line) ==1:
                lists.append(line[0])
            else:
                lists.append(line)
        # pdb.set_trace()
    return lists

def image_list(dir1,txt1,name=None,mode='train'):
    f1 = open(txt1, 'w+')
    path_lists=[]
    for root,dirs,files in os.walk(dir1):

        for file in files:

            if mode =='train':
                if 'tiff' in file and name in file:

                    path = os.path.join(root,file)
                    path_lists.append(path)

            else:
                # pdb.set_trace()
                # find the folder name of the origin dataset, it doesnt contain any words of crystal, loop, and liquor
                if name != 'crystal' and name != 'loop' and name !='liquor' and name!='multi_label':
                    if 'crystal' not in file and 'loop' not in file  and 'liquor' not in file  \
                            and 'multi' not in file  and 'raw' not in file\
                            and 'tiff' in file:
                        path = os.path.join(root, file)
                        path_lists.append(path)
                else:
                    path = os.path.join(root,file)
                    path_lists.append(path)
    path_lists.sort()
    # pdb.set_trace()
    for path in path_lists:
            f1.write(path)
            f1.write("\n")
    f1.close()

def image_whole(txt0,txt1,txt2,txt3,txt4=None,txt5=None,txt6=None,label_only=False):
    # txt1 is the trained dataset
    # txt2 is the crystal mask dataset
    # txt3 is the loop mask dataset
    # txt4 is the liquor mask dataset
    # txt5 is the bubbles mask dataset
    f1 = open(txt0, 'w+')
    path1 = load_lists(txt1)
    path2 = load_lists(txt2)
    path3 = load_lists(txt3)
    if label_only is True:
        iteration_path = path2
    else:
        iteration_path = path1
        assert len(path1) == len(path2)

    assert len(path2) == len(path3)

    if txt4 is not None:
        path4 = load_lists(txt4)
        assert len(path3) == len(path4)
    if txt5 is not None:
        path5 = load_lists(txt5)
        assert len(path4) == len(path5)
    if txt6 is not None:
        path6 = load_lists(txt6)
        assert len(path5) == len(path6)
    for i, dir in enumerate(iteration_path):
        try:
            f1.write(path1[i])
            f1.write('  ')
        except:
            f1.write('end')
            f1.write('  ')

        f1.write(path2[i])
        f1.write('  ')
        f1.write(path3[i])
        f1.write('  ')

        if txt4 is not None:
            f1.write(path4[i])
            f1.write('  ')
        if txt5 is not None:
            f1.write(path5[i])
            f1.write('  ')
        if txt6 is not None:
            f1.write(path6[i])
            f1.write('  ')
        f1.write("\n")
    f1.close()
def test_lists_group(test_root,name='13295',mode='test',multi=True):
    if mode =='test':
        txt1 = os.path.join(test_root,'{}.txt'.format(name))
        txt2 = os.path.join(test_root,'crystal.txt')
        txt3 = os.path.join(test_root, 'loop.txt')
        txt4 = os.path.join(test_root, 'liquor.txt')
        txt5 = os.path.join(test_root, 'multi_label.txt')
        txt0 = os.path.join(test_root,'{}_test.txt'.format(name))
        image_list(test_root, txt5, 'multi')
        image_list(test_root, txt1, name,mode='test')
        image_list(test_root, txt2, 'crystal')
        image_list(test_root, txt3, 'loop')
        image_list(test_root, txt4, 'liquor')

        if multi is False:
            txt5 = None
        image_whole(txt0, txt1, txt2, txt3, txt4, txt5)

        return txt0
    elif mode =='eval':
        txt1 = os.path.join(test_root, '{}.txt'.format(name))
        image_list(test_root, txt1, name, mode='test')
        return  txt1

def train_lists_group(dir1,dir2,txt_dir,whole='whole.txt',bu=False,multi=False):
    txt1 = os.path.join(txt_dir,'trained.txt')
    txt2 = os.path.join(txt_dir,'crystal.txt')
    txt3 = os.path.join(txt_dir, 'loop.txt')
    txt4 = os.path.join(txt_dir, 'liquor.txt')
    txt5 = os.path.join(txt_dir, 'bubbles.txt')
    txt6= os.path.join(txt_dir, 'multi_label.txt')


    txt0 = os.path.join(txt_dir,whole)
    image_list(dir1, txt1, 'whole')
    image_list(dir2, txt2, 'crystal')
    image_list(dir2, txt3, 'loop')
    image_list(dir2, txt4, 'liquor')
    image_list(dir2, txt5, 'bu')
    image_list(dir2, txt6, 'multi')

    if multi is False:
        txt6= None
    if bu is False:
            txt5 = None

    image_whole(txt0, txt1, txt2, txt3, txt4,txt5,txt6,label_only = True)

    return txt0

if __name__ == '__main__':
    dir1 = 'D:/lys/studystudy/phd/absorption correction/dataset/Python3DFirst/blender/laser_slicer-master/testing/batch_ssh/'
    dir = 'D:/lys/studystudy/phd/absorption correction/dataset/13076_13284_13295/13284_tomobar_cropped/13284.txt'
    l = load_lists(dir,mode='eval')
    pdb.set_trace()
