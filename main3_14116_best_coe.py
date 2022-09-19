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

"""
purpose:
--  X-rays scattered by a set atoms produce X-ray radiation in all directions, 
    leading to interferences due to the coherent phase differences between the interatomic vectors 
    that describe the relative position of atoms. 
--  direct lattices with large unit cells produce very close diffracted beams, and vice versa. So for some
    intensities around strong intensity may be omitted
** https://www.xtal.iqfr.csic.es/Cristalografia/index-en.html

several parts:
    1. reconstructing the tomography, aligning the position to real coordinate system
        1) set the position of the origin in the tomography
    2. 
"""

"""  ====Input files====

.refl files (only this one in this code ) and .expt files from DIALS
--.refl files:
    -- loaded by  #from dials.array_family import flex   reflections= flex.reflection_table.from_file(filename)

        -- reflections have many rows and they are dictonary-like class, can be viewed the classes like reflections[0]
            after loading.
        -- examples to change values:
            if want to change the values you have change the whole column:
            c1 = list(np.ones(98563,dtype=np.float64))
            reflections['intensity.sum.variance'] = flex.double(c1) 
            this is because 'intensity.sum.variance', <scitbx_array_family_flex_ext.double object at 0x7f07d0 have a double

            if you want to change a single value:
            reflections[0]={'background.mean': 1} # it changes the background.mean of reflections[0] to 1
            or     row = {"col1": 1000, "col2": 2000, "col3": "hello"}    table[10] = row
            (https://github.com/dials/dials/blob/13944083b0f12cf7d935e008e230954e5171e456/tests/array_family/test_reflection_table.py)
        -- I change it into json file to load it easier.

--.expt files:
    --loaded by json.load
        --    with open(filaname) as f1:
                    data = json.load(f1)
        -- the detailed explanation of each data is  https://dials.github.io/documentation/data_files.html
        -- for I23, the panel numbers are strange, it's curly 
        -- (X, Y, Z) for  goniometer vectors  [1.0, 0.0,0.0],
        -- the lab coordinate of DIALS
            	 * x positive is the gonionmeter, positive (negative) z axis is the beam vector in the raw .expt file (in the dials.show program and my work)
            	 * 
                 *              y
                 *              y
                 *              y
                 *              y
                 *              y
                 *              y
                 *              0 x x x x x x x x 
                 *            z
                 *          z
                 *        z
                 *      z 
                 *
                 */
        -- the coordinate of my system , so rotating the dials system about y axis by 90 degree clockwisely can get my coordinate system.
        Then the normal calculation for phi and theta can be run. The reason why it doesn't invert the y-axis is because when calculate absorption(theta,phi),
        this inversion is involved into the path and absorption calculation  
            	 *
                 *                    z
                 *                   z
                 *                  z
                 *                 z
                 *                z
                 *               z 
                 *              0 x x x x x x x  
                 *              y   
                 *              y
                 *              y
                 *              y
                 *              y
             */
"""

"""
Experiment setup of this code file:
--This code loads the json file generated from .refl file to calculate every absorption for each reflection.
--1. rotating the scattering vectors from dials lab coordinate to fit my coordinate system  M(me) = R(lab)*R(goni)*M(dials)
--2. calculate the theta and phi from the scattering vectors             
    # theta= tan-1(y/-x) at the scattering vector after rotation
    # phi = tan-1(-z/-x) because how phi and my system is define so is tan-1(-z/-x) instead of tan-1(z/-x)
--
"""

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

if __name__ == "__main__":

    """label coordinate loading"""
    rate_list = {'li': 1, 'lo': 2, 'cr': 3, 'bu': 4}
    path_l = ''
    dataset = 14116
    # img_list = np.load(os.path.join(path,'13304_stack.npy'))
    label_list = np.load('./14116_tomobar_cropped_r.npy').astype(np.int8)
    # expt_filaname = '13304_tlys_0p1_4p0keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt'
    refl_filaname = '14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json'
    save_dir = './save_data/{}_best_{}_{}_coe{}'.format(dataset,args.expri,args.angle,args.sqcub)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    zz, yy, xx = np.where(label_list == rate_list['cr'])
    crystal_coordinate = list(zip(zz, yy, xx))  # can be not listise to lower memory usage
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
    coefficient_scaling_factor = 4.133/2.7552
    if args.sqcub == 'sq_cr_li':
    
      mu_cr = 0.0087e3 * (coefficient_scaling_factor**2)    # (unit in mm-1) 16010
      mu_li = 0.0093e3 * (coefficient_scaling_factor**2)
      mu_lo = 0.0115e3
      mu_bu = 0.046e3 
    elif args.sqcub == 'cub_cr_li':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor**3)     # (unit in mm-1) 16010
      mu_li = 0.0093e3 *(coefficient_scaling_factor**3)
      mu_lo = 0.0115e3
      mu_bu = 0.046e3 
    elif args.sqcub == 'lin_cr_li':
      mu_cr = 0.0087e3 * coefficient_scaling_factor    # (unit in mm-1) 16010
      mu_li = 0.0093e3 * coefficient_scaling_factor
      mu_lo = 0.0115e3
      mu_bu = 0.046e3
    elif args.sqcub == 'sq':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor**2)    # (unit in mm-1) 16010
      mu_li = 0.0093e3 * (coefficient_scaling_factor**2)
      mu_lo = 0.0063e3 * ( coefficient_scaling_factor **2)
      mu_bu = 0.0292e3* (coefficient_scaling_factor  **2)
    elif args.sqcub == 'cub':
      mu_cr = 0.0087e3 * (coefficient_scaling_factor**3)     # (unit in mm-1) 16010
      mu_li = 0.0093e3 *(coefficient_scaling_factor**3)
      mu_lo = 0.0063e3 * (coefficient_scaling_factor **3)
      mu_bu = 0.0292e3* (coefficient_scaling_factor **3) 
    elif args.sqcub == 'lin':
      mu_cr = 0.0087e3 * coefficient_scaling_factor    # (unit in mm-1) 16010
      mu_li = 0.0093e3 * coefficient_scaling_factor
      mu_lo = 0.0063e3 * coefficient_scaling_factor 
      mu_bu = 0.0292e3* coefficient_scaling_factor 
    elif args.sqcub == '16010':
      coefficient_scaling_factor =   3.5424/2.7552
      mu_cr = 0.0087e3 * coefficient_scaling_factor  *1.588  # (unit in mm-1) 16010
      mu_li = 0.0093e3 * coefficient_scaling_factor *1.588
      mu_lo = 0.0063e3 * coefficient_scaling_factor *1.588 
      mu_bu = 0.0292e3* coefficient_scaling_factor *1.588 
    elif args.sqcub == 'raw':
      mu_cr = 0.0317e3  # (unit in mm-1) 16010
      mu_li = 0.0303e3
      mu_lo = 0.0115e3
      mu_bu = 0.046e3
    elif args.sqcub == 'new_14116':
      mu_cr = 0.0192e3  # (unit in mm-1) 16010
      mu_li = 0.0184e3
      mu_lo = 0.0162e3
      mu_bu = 0.046e3
    elif args.sqcub == 'my_14116':
      mu_cr = 0.01859e3  # (unit in mm-1) 16010
      mu_li = 0.01816e3
      mu_lo = 0.0162e3
      mu_bu = 0.0427e3
    elif args.sqcub == 'my_14116_auto_hp5_vp1':
      mu_cr = 0.01892e3  # (unit in mm-1) 16010
      mu_li = 0.01882e3
      mu_lo = 0.0162e3
      mu_bu = 0.0399e3
    elif args.sqcub == 'my_14116_auto_hp5_vp1_pp5':
      mu_cr = 0.01892e3  # (unit in mm-1) 16010
      mu_li = 0.0273e3
      mu_lo = 0.0162e3
      mu_bu = 0.0399e3
    elif args.sqcub == 'scale_from_16010':
      coefficient_scaling_factor = 4.133/3.5424
      mu_cr = 0.0103e3 * (coefficient_scaling_factor**3)     # (unit in mm-1) 16010
      mu_li = 0.0130e3 *(coefficient_scaling_factor**3)
      mu_lo = 0.0102e3 * (coefficient_scaling_factor **3)
      mu_bu = 0.0309e3* (coefficient_scaling_factor **3) 
    elif args.sqcub == 'my_14116_auto_hp5_vp1_112loop':
      mu_cr = 0.01892e3  # (unit in mm-1) 16010
      mu_li = 0.01882e3
      mu_lo = 0.0115e3
      mu_bu = 0.0399e3
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
    #offset = ( - 10.095 -(77.357)) / 180 * np.pi
    offset = -90 / 180 * np.pi

    #offset = ( - 5.095 -(77.357)) / 180 * np.pi
    # -2.994  is the snapshot degree,
    # the rotation matrix has adjusted the rotation of the lab coordinate
    # so here, just use the origin lab data
    # for the -87.278, is the inclined angle to the left axis in the tomography slice
    # if rotate to the left axis, then it's anticlockwise to the lab, so it's negative
    # if rotate to the right axis, then it's clockwise to the lab, so it's positive

   
    # multiprocesss
    # p = multiprocessing.Pool(4)
    #
    # multi = multi_pro(data,crystal_coordinate,coe,rate_list,label_list,offset)
    # result= p.map(multi,range(len(data[0:10])))
    #
    # with open("./data/13304_refl.json", "w") as fz:  # Pickling
    #     json.dump(result, fz, indent=2)
    # t2=time.time()
    # print('it spends {}'.format(t2 - t1))
    # print('Finish!!!!')

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
        rotation_frame_angle += offset
        # rotation_frame_angle = rotation_frame_angle - offset  # offset is the starting angle
        if rotation_frame_angle < 0:
            if rotation_frame_angle < 2*np.pi:
                  rotation_frame_angle = 4 * np.pi + rotation_frame_angle
            else:
                  rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2*np.pi:
            if  rotation_frame_angle > 4*np.pi:
                  rotation_frame_angle = rotation_frame_angle - 4 * np.pi
            else:
                  rotation_frame_angle = rotation_frame_angle - 2 * np.pi

        assert rotation_frame_angle <= 2 * np.pi

        rotation_matrix_frame = np.array([[1, 0, 0],
                                          [0, np.cos(rotation_frame_angle), np.sin(rotation_frame_angle)],
                                          [0, -np.sin(rotation_frame_angle), np.cos(rotation_frame_angle)]])
        #  rotate the x-ray beam about x-axis omega degree clockwisely

        # final_rotation = np.dot(rotation_matrix_lab, rotation_matrix_frame)
       #rotated_s1 = np.dot(final_rotation, scattering_vector)
       #rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
#       if rotated_s1[0] == 0:
#           # tan-1(y/-x) at the scattering vector after rotation
#           theta = np.arctan(rotated_s1[1] / (-rotated_s1[0] + 0.001))
#           # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#           phi = np.arctan(rotated_s1[2] / (rotated_s1[0] + 0.001))
#       else:
#           if rotated_s1[0] < 0:
#               theta = np.arctan(rotated_s1[1] / (-rotated_s1[0]))  # tan-1(y/-x)
#           else:
#               if rotated_s1[1] > 0:
#                   theta = np.pi - np.arctan(rotated_s1[1] / rotated_s1[0])  # tan-1(y/-x)
#               else:
#                   theta = - np.pi - np.arctan(rotated_s1[1] / rotated_s1[0])  # tan-1(y/-x)
#           phi = np.arctan(rotated_s1[2] / (rotated_s1[0]))  # tan-1(-z/-x)

        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
        if rotated_s1[2] == 0:
            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
            theta = np.arctan(rotated_s1[1] / (-np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2) + 0.001))
            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
            phi = np.arctan(rotated_s1[0] / (rotated_s1[2] + 0.001))
        else:
            if rotated_s1[2] < 0:
                theta = np.arctan(rotated_s1[1] / np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2) )  # tan-1(y/-x)
                phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))
            else:
                if rotated_s1[1] > 0:
                    theta = np.pi - np.arctan(rotated_s1[1] /np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2))  # tan-1(y/-x)

                else:
                    theta = - np.pi - np.arctan(rotated_s1[1] / np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2))  # tan-1(y/-x)
                phi = np.arctan(rotated_s1[0] / (rotated_s1[2]))  # tan-1(-z/-x)
#        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
#        if rotated_s1[2] == 0:
#            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#            theta = np.arctan(rotated_s1[1] / (np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2) + 0.001))
#            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#            phi = np.arctan(-rotated_s1[0] / (rotated_s1[2] + 0.001))
#        else:
#            if rotated_s1[2] >= 0:
#                theta = np.arctan(rotated_s1[1] / np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2) )  # tan-1(y/-x)
#                #phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))
#                phi = np.arctan(rotated_s1[0] / np.abs(rotated_s1[2]))
#            else:
#                if rotated_s1[1] > 0:
#                    theta = np.pi - np.arctan(rotated_s1[1] /np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2))  # tan-1(y/-x)
#
#                else:
#                    theta = -(np.pi - np.abs( np.arctan(rotated_s1[1] / np.sqrt( rotated_s1[0]**2+ rotated_s1[2]**2)) ) )# tan-1(y/-x)
#                #phi = np.arctan(rotated_s1[0] / (rotated_s1[2]))  # tan-1(-z/-x)
#                phi = -np.arctan(-rotated_s1[0] / np.abs(rotated_s1[2]))

#        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
#        """" 1.first(test_f_1) face to the detector, x is towards right, 
#             y is towards top, z is towards out of the screen """
#        if rotated_s1[2] == 0:
#            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
#            theta = np.arctan(rotated_s1[1] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2) + 0.001))
#            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
#            phi = np.arctan(rotated_s1[0] / (-rotated_s1[2] + 0.001))
#        else:
#            if rotated_s1[2] < 0:
#                theta = np.arctan(rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#                phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))
#            else:
#                if rotated_s1[1] > 0:
#                    theta = np.pi - np.arctan(rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#                else:
#                    theta = - np.pi - np.arctan(rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
#                phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))  # tan-1(-z/-x)

        # omega = np.abs(np.arctan(np.cos(phi) * np.tan(theta)))
        absorp = np.empty(len(coordinate_list))
#        errors_0 = [ rot_error_the, rot_error_phi]
#        errors_1 = np.empty((len(coordinate_list), 2))
#        errors_2 = np.empty((len(coordinate_list), 3))

        for k, index in enumerate(coordinate_list):
            coord = crystal_coordinate[index]
            face_2 = which_face_2(coord, shape, theta, phi)  # 1s
            face_1 = which_face_1_anti(coord, shape, rotation_frame_angle)  # 0.83
            path_1 = cal_coord_1_anti(rotation_frame_angle, coord, face_1, shape, label_list)  # 37
            path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list,full_iteration=False)  # 16
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
#        print(errors_0)
#        print(np.mean(errors_1, axis=0))
#        print(np.mean(errors_2, axis=0))
        print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                                  low + len(
                                                                                                      selected_data),
                                                                                                  theta * 180 / np.pi,
                                                                                                  phi * 180 / np.pi,
                                                                                                  rotation_frame_angle * 180 / np.pi,
                                                                                                  absorp.mean()))
        # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
  
        corr.append(absorp.mean())
        
#        if i ==0:
#            rot_errors = [errors_0]
#            path_1_errors = [np.abs(np.mean(errors_1, axis=0))]
#            path_2_errors = [np.abs(np.mean(errors_2, axis=0))]
#        else:
#            rot_errors = np.concatenate((rot_errors, [errors_0]), axis=0)
#            path_1_errors = np.concatenate((path_1_errors, [np.abs(np.mean(errors_1, axis=0))]), axis=0)
#            path_2_errors = np.concatenate((path_2_errors, [np.abs(np.mean(errors_2, axis=0))]), axis=0)
        t2 = time.time()

        print('it spends {}'.format(t2 - t1))
        dict_corr.append({'index': low + i, 'miller_index': miller_index,
                          'intensity': intensity, 'corr': absorp.mean(), 'lp': lp})
        del absorp
        if i % 500 == 1:
            np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
            with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
                json.dump(corr, fz, indent=2)

            with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
                json.dump(dict_corr, f1, indent=2)
#            np.save(os.path.join(save_dir, "{}_refl_rot_error_{}.npy".format(dataset, up)), rot_errors)
#            np.save(os.path.join(save_dir, "{}_refl_path_1_error_{}.npy".format(dataset, up)), path_1_errors)
#            np.save(os.path.join(save_dir, "{}_refl_path_2_error_{}.npy".format(dataset, up)), path_2_errors)
    
    np.save( os.path.join(save_dir, "{}_path_lengths_{}.npy".format(dataset, up)),  path_length_arr  )
    with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
        json.dump(corr, fz, indent=2)

    with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
        json.dump(dict_corr, f1, indent=2)
#     np.save(os.path.join(save_dir, "{}_refl_rot_error_{}.npy".format(dataset, up)), rot_errors)
#     np.save(os.path.join(save_dir, "{}_refl_path_1_error_{}.npy".format(dataset, up)), path_1_errors)
#     np.save(os.path.join(save_dir, "{}_refl_path_2_error_{}.npy".format(dataset, up)), path_2_errors)
    print('Finish!!!!')








