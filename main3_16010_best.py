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
from utils_lite import *


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
    "--rotate-angle",
    type=int,
    default=0,
    help="goinometer rotate-angle",
)
parser.add_argument(
    "--dataset",
    type=int,
    default=16010,
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
    "--coordset",
    type=str,
    default=0,
    help="coordinate setting",
)
parser.add_argument(
    "--sqcub",
    type=str,
    default='sq',
    help="coordinate setting",
)



global args
args = parser.parse_args()


if __name__ == "__main__":

    """label coordinate loading"""
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
    path_l = ''
    dataset = 16010
    # img_list = np.load(os.path.join(path,'13304_stack.npy'))
    label_list = np.load('./16010_tomobar_cropped_f.npy').astype(np.int8)
    # expt_filaname = '13304_tlys_0p1_4p0keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt'
    refl_filaname = '16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json'
    #refl_filaname = 'scaled_0.refl.json'
    save_dir = './save_data/{}_best_full_iteration_{}_{}_coordset_{}_coe{}'.format(dataset,args.expri,args.angle,args.coordset,args.sqcub)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    zz, yy, xx = np.where(label_list == rate_list['cr'])
    crystal_coordinate = np.stack((zz,yy,xx),axis=1)  
    origin = np.array([int(np.round((np.max(zz) + np.min(zz)) / 2)),
                       int(np.round((np.max(yy) + np.min(yy)) / 2)),
                       int(np.round((np.max(xx) + np.min(xx)) / 2))])

    """tomography setup """
    pixel_size = 0.3e-3  # it means how large for a pixel of tomobar in real life
    tomo_dep = pixel_size * label_list.shape[2]  # x, depth of the tomo if it's normal to the wavevector
    tomo_h = pixel_size * label_list.shape[1]  # y
    tomo_w = pixel_size * label_list.shape[0]  # z

    """experiment in lab hyperparameter"""
    lab_origin = origin  # lab origin is the centre of the crystal
    # wavelength =3.099600e-7  # unit in  angstrom 10e-10m adn 10e-7 in mm
    coefficient_scaling_factor =   (3.5424/2.7552)
    #coefficient_scaling_factor = 1
    if args.sqcub == 'sq':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor) **2   # (unit in mm-1) 16010
      mu_li = 0.0093e3 * (coefficient_scaling_factor) **2
      mu_lo = 0.0063e3 * (coefficient_scaling_factor) **2
      mu_bu = 0.0292e3 * (coefficient_scaling_factor) **2
    elif args.sqcub == 'cub':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor) **3   # (unit in mm-1) 16010
      mu_li = 0.0093e3 * (coefficient_scaling_factor) **3
      mu_lo = 0.0063e3 * (coefficient_scaling_factor) **3
      mu_bu = 0.0292e3 * (coefficient_scaling_factor) **3
    elif args.sqcub == '14116':
      mu_cr = 0.0317e3 / 1.588  # (unit in mm-1) 16010
      mu_li = 0.0303e3 / 1.588 
      mu_lo = 0.0115e3 / 1.588 
      mu_bu = 0.046e3 / 1.588 
    elif args.sqcub == 'raw':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor)   # (unit in mm-1) 16010
      mu_li = 0.0093e3 * (coefficient_scaling_factor) 
      mu_lo = 0.0063e3 * (coefficient_scaling_factor) 
      mu_bu = 0.0292e3 * (coefficient_scaling_factor) 
    elif args.sqcub == 'new_16010':
      mu_cr = 0.0103e3 # (unit in mm-1) 16010
      mu_li = 0.0130e3 
      mu_lo = 0.0102e3 
      mu_bu = 0.0309e3 
    elif args.sqcub == 'my_16010':
      mu_cr = 0.01174e3 # (unit in mm-1) 16010
      mu_li = 0.01204e3 
      mu_lo = 0.0102e3 
      mu_bu = 0.02701e3 
    coe = {'mu_li': mu_li, 'mu_lo': mu_lo, "mu_cr": mu_cr}

    t1 = time.time()
    shape = label_list.shape
    sampling = 2000
    seg = int(np.round(len(crystal_coordinate) / sampling))
    # coordinate_list =range(0,len(crystal_coordinate),seg)  # sample points from the crystal pixel
    coordinate_list = np.linspace(0, len(crystal_coordinate), num=seg, endpoint=False, dtype=int)
    with open(refl_filaname) as f1:
        data = json.load(f1)
    print('The total size of the dataset is {}'.format(len(data)))
    corr = []
    dict_corr = []
    # path_2_errors = np.array([[0, 0, 0]])
    # offset = -np.min(np.array([literal_eval(i['xyzobs.mm.value'])[2] for i in data]))
    offset = ( - 87.278 -(-2.994)) / 180 * np.pi
    offset = float(args.angle) / 180 * np.pi
    print('the offset angle is {}'.format(offset))
    # single processs
    low = args.low
    up = args.up


    if up == -1:
        selected_data = data[low:]
    else:
        selected_data = data[low:up]

    del data
    coefficients = mu_li, mu_lo, mu_cr, mu_bu

    for i, row in enumerate(selected_data):
        intensity = float(row['intensity.sum.value'])
        scattering_vector = literal_eval(row['s1'])  # all are in x, y , z in the origin dials file
        miller_index = row['miller_index']
        lp = row['lp']

        rotation_matrix_lab = np.array([[0, 0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]])  # rotate the coordinate system about y-axis 90 degree clockwisely
        # so the right top should be -sin(theta), bottom left is sin(theta)
        rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]
        if args.coordset ==0:
            rotation_frame_angle += offset  # offset is the starting angle
        else:
            rotation_frame_angle += 0
        
        if rotation_frame_angle < 0:
            rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2*np.pi:
            rotation_frame_angle = rotation_frame_angle - 2 * np.pi

        assert rotation_frame_angle <= 2 * np.pi

        rotation_matrix_frame = np.array([[1, 0, 0],
                                          [0, np.cos(rotation_frame_angle), np.sin(rotation_frame_angle)],
                                          [0, -np.sin(rotation_frame_angle), np.cos(rotation_frame_angle)]])
        #  rotate the x-ray beam about x-axis omega degree clockwisely

        absorp = np.empty(len(coordinate_list))
        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
        xray=np.array([0,0,-1])
        xray=np.dot(rotation_matrix_frame,xray)
        
        if args.coordset == "0":
        
            theta,phi=dials_2_thetaphi(rotated_s1)
            theta_1,phi_1=dials_2_thetaphi(xray,L1=True)
        elif args.coordset == "11":

            theta,phi=dials_2_thetaphi_11(rotated_s1)
            theta_1,phi_1=dials_2_thetaphi_11(xray,L1=True)
        elif args.coordset == "22":
        
            theta,phi=dials_2_thetaphi_22(rotated_s1)
            theta_1,phi_1=dials_2_thetaphi_22(xray,L1=True)
            
        for k, index in enumerate(coordinate_list):
            coord = crystal_coordinate[index]
            #face_2 = which_face_2(coord, shape, theta, phi)  # 1s

            #face_1 = which_face_1_anti(coord, shape, rotation_frame_angle)  # 0.83
            face_1 = which_face_2(coord, shape, theta_1, phi_1) 
            face_2 = which_face_2(coord, shape, theta, phi) 
            
            path_1 = cal_coord_2(theta_1,phi_1,coord,face_1,shape,label_list) # 37
#            face_2 = which_face_matrix(coord,rotated_s1,shape)
#            face_1 = which_face_matrix(coord,xray,shape,exit=False)
#            path_1 = cal_coord_1_anti(rotation_frame_angle, coord, face_1, shape, label_list)
            path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list)  # 16
            numbers = cal_num(coord, path_1, path_2, theta, rotation_frame_angle)  # 3.5s
            if k ==0:
                path_length_arr_single = np.expand_dims( np.array(numbers),axis=0 )
            else:
                
                path_length_arr_single = np.concatenate( ( path_length_arr_single,np.expand_dims( np.array(numbers),axis=0 ))   ,axis =0 )
            absorption = cal_rate(numbers, coefficients, pixel_size)
            absorp[k] = absorption
 
        if i ==0:
          path_length_arr= np.expand_dims(path_length_arr_single,axis=0 )
        else:
          path_length_arr= np.concatenate( ( path_length_arr,np.expand_dims(path_length_arr_single,axis=0 ))   ,axis =0 )
            
#            path_12=iterative_bisection(theta_1,phi_1,coord,face_1,label_list)
#            path_22=iterative_bisection(theta, phi,coord,face_2,label_list)
#            numbers_2 = cal_num22(coord,path_12,path_22,theta,rotation_frame_angle)
#            absorption = cal_rate(numbers_2, coefficients, pixel_size)
#            absorp[k] = absorption

            # if error_theta * 180 / np.pi >2:
            #     pdb.set_trace()
        print(absorp.mean())
        print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                  low + len(
                                                                                                      selected_data),
                                                                                                  theta * 180 / np.pi,
                                                                                                  phi * 180 / np.pi,
                                                                                                  rotation_frame_angle * 180 / np.pi,
                                                                                                  absorp.mean()))
        # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type

        corr.append(absorp.mean())

        t2 = time.time()

        print('it spends {}'.format(t2 - t1))
        dict_corr.append({'index': low + i, 'miller_index': miller_index,
                          'intensity': intensity, 'corr': absorp.mean(), 'lp': lp})
        if i % 500 == 1:
            np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
            with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
                json.dump(corr, fz, indent=2)

            with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
                json.dump(dict_corr, f1, indent=2)

    np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
    with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
        json.dump(corr, fz, indent=2)

    with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
        json.dump(dict_corr, f1, indent=2)
    print('Finish!!!!')








