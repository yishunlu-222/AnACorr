import os
import json
# import pickle
# from matplotlib import pyplot as plt
# from multiprocessing import Process
# import multiprocessing
import time
import pdb
import numpy as np
# from dials.array_family import flex
from ast import literal_eval
import argparse
from utils import *
from unit_test import *
from sys import getsizeof
import resource
import  gc
import shutil
import sys
# ===========================================
#        Parse the argument
# ===========================================

parser = argparse.ArgumentParser(description="multiprocessing for batches")

parser.add_argument(
    "--low",
    type=int,
    default=0,
    help="the starting point of the batch",
)
parser.add_argument(
    "--up",
    type=int,
    default=-1,
    help="the ending point of the batch",
)

parser.add_argument(
    "--save",
    default=True,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--angle",
    type=str,
    default=-90,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--expri",
    type=str,
    default=0,
    help="dataset number default is 13304",
)

parser.add_argument(
    "--save-index",
    type=int,
    default=1,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--dataset",
    type=int,
    default=16010,
    help="1 is true, 0 is false",
)

global args
args = parser.parse_args()

"""experiment detector hyperparameter

"""
# with open(expt_filaname) as f:
#     expt = json.load(f)

# p_W_detector = expt['detector'][0]['panels'][0]['image_size'][0]
# p_H_detector = expt['detector'][0]['panels'][0]['image_size'][1]
# x_pixel_size_detector = expt['detector'][0]['panels'][0]['pixel_size'][0]  # in mm
# y_pixel_size_detector_each = expt['detector'][0]['panels'][0]['pixel_size'][1]  # in mm
# # W_detector = p_W_detector * x_pixel_size_detector
# # H_detector = p_H_detector * y_pixel_size_detector
# absorption_map = np.empty((len(expt['detector'][0]['panels']), p_H_detector, p_W_detector))

"""loading integrated data from dials.refl"""
"""
{'background.mean': 3.0067946910858154,
'background.sum.value': 93.21065521240234, 
 'background.sum.variance': 118.78172302246094, 
 'bbox': (1310, 1322, 1520, 1532, 0, 1), 
 'd': 4.050291949582959, 
 'entering': False, 
 'flags': 1048833, 
 'id': 0, 
 'imageset_id': 0, 
 'intensity.prf.value': 0.0, 
 'intensity.prf.variance': -1.0, 
 'intensity.sum.value': 7.789344787597656, 
 'intensity.sum.variance': 126.5710678100586, 
 'lp': 0.30011041432316177, 
 'miller_index': (1, -5, 16), 
 'num_pixels.background': 113, 
 'num_pixels.background_used': 113, 
 'num_pixels.foreground': 31,
 'num_pixels.valid': 144, 
 'panel': 0, 
 'partial_id': 1, 
 'partiality': 0.0, 
 'profile.correlation': 0.0,
 'qe': 0.9300963202028352, 
 's1': (0.03337449526869561, -0.24160215159383425, -0.7687887369744076),
 'xyzcal.mm': (226.38154049755676, 262.4721810886861, -0.01511517970316119), 
 'xyzcal.px': (1316.2015583386494, 1526.169962473035, -1.7320720071458813), 
 'xyzobs.mm.value': (226.4564177359127, 262.41171771334706, 0.0043633227029589395), 
 'xyzobs.mm.variance': (0.005281526683090885, 0.005847856615085115, 6.346196245556437e-06), 
 'xyzobs.px.value': (1316.6371377429057, 1525.8183361999768, 0.4999999510663235),
 'xyzobs.px.variance': (0.17852645629701475, 0.19766957189984835, 0.08333333333333345), 
 'zeta': 0.9902352872689145}
"""


def clear():
    for key, value in globals().items():

        if callable(value) or value.__class__.__name__ == "module":
            continue

        del globals()[key]


def kappa_phi_rotation_matrix(axis, theta, raytracing=True):
    """
    Euler-Rodrigues formula
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    if raytracing:
      axis = -axis
    else:
      axis = axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    # pdb.set_trace()
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def kp_rotation(axis,theta, raytracing=True):
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param axis:
    :param theta:
    :return:
    """

    x,y,z = axis
    c =np.cos(theta)
    s = np.sin(theta)
    first_row = np.array([ c + (x**2)*(1-c), x*y*(1-c) - z*s, y*s + x*z*(1-c)  ])
    seconde_row = np.array([z*s + x*y*(1-c),  c + (y**2)*(1-c) , -x*s + y*z*(1-c) ])
    third_row = np.array([ -y*s + x*z*(1-c), x*s + y*z*(1-c), c + (z**2)*(1-c)  ])
    matrix = np.stack(( first_row, seconde_row, third_row), axis = 0)
    return matrix


def rodrigues(n,theta):
    """
    https://mathworld.wolfram.com/RodriguesRotationFormula.html
    matrix calculation 
    :param axis:
    :param theta:
    :return:
    """
    def S(n):
        Sn = np.array([[0,-n[2],n[1]],
                       [n[2],0,-n[0]],
                       [-n[1],n[0],0]])
        return Sn
    Sn = S(n)
    R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn,Sn)

    return np.mat(R)


if __name__ == "__main__":

    """label coordinate loading"""
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
    path_l = ''
    dataset = 16010
    # img_list = np.load(os.path.join(path,'13304_stack.npy'))
    label_list = np.load('./16010_tomobar_cropped_f.npy').astype(np.int8)
    refl_filaname = '16010_ompk_10_3p5keV_km70_pp120_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json'
    expt_filaname = '16010_ompk_10_3p5keV_km70_pp120_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt.json'   # only contain axes
    save_dir = './save_data/{}_best_km70_pp120_{}_{}_(okp)v_nor'.format(dataset,args.expri,args.angle)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    shutil.copy(sys.argv[0],  save_dir)
    shutil.copy('./utils.py',  save_dir)
    zz, yy, xx = np.where(label_list == rate_list['cr'])  # this line occupies 1GB, why???
    #crystal_coordinate = zip(zz, yy, xx)  # can be not listise to lower memory usage
    crystal_coordinate = np.stack((zz,yy,xx),axis=1) 
    # length_crystal_coordinate=len(list(crystal_coordinate))
    del zz, yy, xx  #
    gc.collect()
    """tomography setup """
    pixel_size = 0.3e-3  # it means how large for a pixel of tomobar in real life

    """experiment in lab hyperparameter"""
    # wavelength =3.099600e-7  # unit in  angstrom 10e-10m adn 10e-7 in mm
    """ coefficients are calculated at 4.5kev but the diffraction experiment is in 3.5kev so scaling is needed, but don't know linear is good """
    coefficient_scaling_factor =   3.5424/2.7552
    #coefficient_scaling_factor = 1
    mu_cr = 0.0087e3 * coefficient_scaling_factor    # (unit in mm-1) 16010
    mu_li = 0.0093e3 * coefficient_scaling_factor
    mu_lo = 0.0063e3 * coefficient_scaling_factor  
    mu_bu = 0.0292e3 * coefficient_scaling_factor 
    coe = {'mu_li': mu_li, 'mu_lo': mu_lo, "mu_cr": mu_cr}
    #
    t1 = time.time()
    shape = label_list.shape
    sampling = 2000
    seg = int(np.round(len(crystal_coordinate) / sampling))
    # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel
    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    with open(expt_filaname) as f2:
        axes = json.load(f2)
    with open(refl_filaname) as f1:
        data = json.load(f1)
    print('The total size of the dataset is {}'.format(len(data)))
    corr = []
    dict_corr = []
    # path_2_errors = np.array([[0, 0, 0]])
    # offset = -np.min(np.array([literal_eval(i['xyzobs.mm.value'])[2] for i in data]))
#    offset = ( - 87.278 -(-2.994)) / 180 * np.pi
    offset = float(args.angle) / 180 * np.pi
#    print('the offset angle is {}'.format(offset))



    # single processs
    low = args.low
    up = args.up

    if up == -1:
        select_data = data[low:]
    else:
        select_data = data[low:up]

    del data
    coefficients = mu_li, mu_lo, mu_cr, mu_bu
    
    


    
    U_matrix=np.reshape( np.array(axes[5]) , (3,3) )
    B_matrix=np.reshape( np.array(axes[6]) , (3,3) )
    A_matrix=np.reshape( np.array(axes[7]) , (3,3) )

    U_matrix_anti = np.linalg.inv(U_matrix)
    B_matrix_anti = np.linalg.inv(B_matrix)
    A_matrix_anti = np.linalg.inv(A_matrix)


    kappa_axis=np.array(axes[2])
    kappa=axes[4][1]/180*np.pi
    kappa_matrix = kp_rotation(kappa_axis, kappa)
    
    phi_axis=np.array(axes[1])
    phi=axes[4][0]/180*np.pi
    phi_matrix = kp_rotation(phi_axis, phi)
    #F=np.dot(phi_matrix_anti,kappa_matrix_anti  )    #https://dials.github.io/documentation/conventions.html#equation-diffractometer
    
    omega_axis=np.array(axes[3])
    F = np.dot(kappa_matrix , phi_matrix )   # phi is the most intrinsic rotation, then kappa 
    
    for i, row in enumerate(select_data):
        intensity = float(row['intensity.sum.value'])
        scattering_vector = literal_eval(row['s1'])  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']
        lp = row['lp']

        rotation_matrix_lab = np.array([[0, 0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]])  # rotate the coordinate system about y-axis 90 degree clockwisely
        # so the right top should be -sin(theta), bottom left is sin(theta)
        rotation_frame_angle_0 = literal_eval(row['xyzobs.mm.value'])[2]
        rotation_frame_angle = rotation_frame_angle_0
        
        #rotation_frame_angle += offset  # offset is the starting angle
        if rotation_frame_angle < 0:
            if rotation_frame_angle < 2 * np.pi:
                rotation_frame_angle = 4 * np.pi + rotation_frame_angle
            else:
                rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2 * np.pi:
            if rotation_frame_angle > 4 * np.pi:
                rotation_frame_angle = rotation_frame_angle - 4 * np.pi
            else:
                rotation_frame_angle = rotation_frame_angle - 2 * np.pi

        assert rotation_frame_angle <= 2 * np.pi

        offset_matrix = np.array([[1, 0, 0],
                                          [0, np.cos(offset), np.sin(offset)],
                                          [0, -np.sin(offset), np.cos(offset)]])
        rotation_matrix_frame_2 = kp_rotation(omega_axis, rotation_frame_angle) 
        
        rotation_matrix_frame_1 = np.array([[1, 0, 0],
                                          [0, np.cos(rotation_frame_angle), np.sin(rotation_frame_angle)],
                                          [0, -np.sin(rotation_frame_angle), np.cos(rotation_frame_angle)]])
                                          
        rotation_matrix_frame = np.array([[1, 0, 0],
                                          [0, np.cos(rotation_frame_angle), -np.sin(rotation_frame_angle)],
                                          [0, np.sin(rotation_frame_angle), np.cos(rotation_frame_angle)]])                                          
        
        #rotation_matrix_frame = np.dot(rotation_matrix_frame_1,F)
        
        #rotation_matrix_frame = np.dot(offset_matrix,rotation_matrix_frame_1)
        rotation_matrix_frame = np.dot(rotation_matrix_frame_2,F)
        rotation_matrix_frame = np.transpose(rotation_matrix_frame)
        #rotation_matrix_frame = np.dot(offset_matrix,rotation_matrix_frame)
        
        
        absorp = np.empty(len(coordinate_list))
        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
        xray=np.array([0,0,-1])
        xray=np.dot(rotation_matrix_frame,xray)
        theta,phi=dials_2_thetaphi_22(rotated_s1)
        theta_1,phi_1=dials_2_thetaphi_22(xray,L1=True)

        for k, index in enumerate(coordinate_list):
            coord = crystal_coordinate[index]
            face_1 = which_face_2(coord, shape, theta_1, phi_1) 
            face_2 = which_face_2(coord, shape, theta, phi) 
            path_1 = cal_coord_2(theta_1,phi_1,coord,face_1,shape,label_list) # 37
            
#            face_2 = which_face_matrix(coord,rotated_s1,shape)
#            face_1 = which_face_matrix(coord,xray,shape,exit=False)
#            path_1 = cal_coord_1_anti(rotation_frame_angle, coord, face_1, shape, label_list)
            
            path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list)  # 16
            numbers = cal_num(coord, path_1, path_2, theta, rotation_frame_angle)  # 3.5s
            absorption = cal_rate(numbers, coefficients, pixel_size)
            absorp[k] = absorption
            # if error_theta * 180 / np.pi >2:

        print(absorp.mean())

        print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                  low + len(
                                                                                                      select_data),
                                                                                                  theta * 180 / np.pi,
                                                                                                  phi * 180 / np.pi,
                                                                                                  rotation_frame_angle * 180 / np.pi,
                                                                                                  absorp.mean()))
        # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
        
        t2 = time.time()
        print('it spends {}'.format(t2 - t1))
        if args.save_index ==1:
          corr.append(absorp.mean())
          print('saving the results')

            
          
          dict_corr.append({'index': low + i, 'miller_index': miller_index,
                            'theta': theta * 180 / np.pi, 'corr': absorp.mean(), 'phi':  phi * 180 / np.pi, 'omega': rotation_frame_angle_0 * 180 / np.pi})
          if i % 500 == 1:
              with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
                  json.dump(corr, fz, indent=2)
  
              with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
                  json.dump(dict_corr, f1, indent=2)


    if args.save_index==1:
      with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
          json.dump(corr, fz, indent=2)
  
      with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
          json.dump(dict_corr, f1, indent=2)

    print('Finish!!!!')








