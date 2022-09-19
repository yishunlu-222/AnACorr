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
np.set_printoptions(suppress=True)

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
    "--angle",
    type=int,
    default=0,
    help="goinometer rotate-angle",
)
parser.add_argument(
    "--dataset",
    type=int,
    default=13304,
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
    "--li-augment",
    type=float,
    default=1,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--lo-augment",
    type=float,
    default=1,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--cr-augment",
    type=float,
    default=1,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--bu-augment",
    type=float,
    default=1,
    help="1 is true, 0 is false",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default='./',
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



if __name__ == "__main__":

    """label coordinate loading"""
    rate_list = {'li': 1, 'lo': 2, 'cr': 3 , 'bu' : 4}
    path_l = ''
    dataset = 14116
    label_list = np.load('./14116_tomobar_cropped_r.npy').astype(np.int8)
    refl_filaname = '14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl.json'
    save_dir = './save_data/{0}/{1}_test_{2}_{3}_{4}_{5}_{6}_{7}'.format(args.save_dir,dataset,
                                                                args.expri,
                                                                args.angle,
                                                                args.li_augment,
                                                                args.lo_augment,
                                                                args.cr_augment,
                                                                args.bu_augment)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
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
    print(args.li_augment)
    print(args.lo_augment)
    print(args.cr_augment)
    """experiment in lab hyperparameter"""
    lab_origin = origin  # lab origin is the centre of the crystal
    # wavelength =3.099600e-7  # unit in  angstrom 10e-10m adn 10e-7 in mm
    # mu_cr = 0.0136e3*args.cr_augment # (unit in mm-1) 13304
    # mu_li = 0.0232e3*args.li_augment
    # mu_lo = 0.0072e3*args.lo_augment
    # mu_bu = 0
    mu_cr = 0.0317e3*args.cr_augment  # (unit in mm-1) 16010
    mu_li = 0.0303e3*args.li_augment
    mu_lo = 0.0115e3*args.lo_augment
    mu_bu = 0.046e3*args.bu_augment
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

    # offset = -np.min(np.array([literal_eval(i['xyzobs.mm.value'])[2] for i in data]))

    offset = (- 10.095 - (77.357)) / 180 * np.pi
    """The offset is offset =(121.707-(-34.47)) / 180 * np.pi -34.47 is rotating the image anti-clockwisely 
    and 121.707 is the angle of the parallel fitting image """

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
        scattering_vector = literal_eval(row['s1'])
        miller_index = row['miller_index']
        lp = row['lp']

        rotation_matrix_lab = np.array([[0, 0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]])  # rotate the coordinate system about y-axis 90 degree clockwisely
                                                    # so the right top should be -sin(theta), bottom left is sin(theta)
        rotation_frame_angle = literal_eval(row['xyzobs.mm.value'])[2]

        rotation_frame_angle += offset  # offset is the starting angle

        if rotation_frame_angle < 0:
            rotation_frame_angle = 2 * np.pi + rotation_frame_angle
        if rotation_frame_angle > 2*np.pi:
            rotation_frame_angle = rotation_frame_angle - 2 * np.pi
        assert rotation_frame_angle <= 2 * np.pi
        rotation_matrix_frame = np.array([[1, 0, 0],
                                          [0, np.cos(rotation_frame_angle), np.sin(rotation_frame_angle)],
                                          [0, -np.sin(rotation_frame_angle), np.cos(rotation_frame_angle)]])
        #  rotate the x-ray beam about x-axis omega degree clockwisely
        # final_rotation = np.dot(rotation_matrix_lab, rotation_matrix_frame)
        # rotated_s1 = np.dot(final_rotation, scattering_vector)
        #
        # if rotated_s1[0] == 0:
        #     # tan-1(y/-x) at the scattering vector after rotation
        #     theta = np.arctan(rotated_s1[1] / (-rotated_s1[0] + 0.001))
        #     # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
        #     phi = np.arctan(rotated_s1[2] / (rotated_s1[0] + 0.001))
        # else:
        #     if rotated_s1[0] < 0:
        #         theta = np.arctan(rotated_s1[1] / (-rotated_s1[0]))  # tan-1(y/-x)
        #     else:
        #         if rotated_s1[1] > 0:
        #             theta = np.pi - np.arctan(rotated_s1[1] / rotated_s1[0])  # tan-1(y/-x)
        #         else:
        #             theta = - np.pi - np.arctan(rotated_s1[1] / rotated_s1[0])  # tan-1(y/-x)
        #     phi = np.arctan(rotated_s1[2] / (rotated_s1[0]))  # tan-1(-z/-x)

        rotated_s1 = np.dot(rotation_matrix_frame, scattering_vector)
        if rotated_s1[2] == 0:
            # tan-1(y/-x) at the scattering vector after rotation np.arctan(y/np.sqrt( x**2+ z**2))
            theta = np.arctan(rotated_s1[1] / (-np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2) + 0.001))
            # tan-1(-z/-x) because how phi and my system are defined so is tan-1(-z/-x) instead of tan-1(z/-x)
            phi = np.arctan(rotated_s1[0] / (rotated_s1[2] + 0.001))
        else:
            if rotated_s1[2] < 0:
                theta = np.arctan(rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
                phi = np.arctan(rotated_s1[0] / (-rotated_s1[2]))
            else:
                if rotated_s1[1] > 0:
                    theta = np.pi - np.arctan(
                        rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)

                else:
                    theta = - np.pi - np.arctan(
                        rotated_s1[1] / np.sqrt(rotated_s1[0] ** 2 + rotated_s1[2] ** 2))  # tan-1(y/-x)
                phi = np.arctan(rotated_s1[0] / (rotated_s1[2]))  # tan-1(-z/-x)
        # omega = np.abs(np.arctan(np.cos(phi) * np.tan(theta)))

        absorp = np.empty(len(coordinate_list))

        for k, index in enumerate(coordinate_list):
            coord = crystal_coordinate[index]
            face_2 = which_face_2(coord, shape, theta, phi)  # 1s
            face_1 = which_face_1_anti(coord, shape, rotation_frame_angle)  # 0.83
            path_1 = cal_coord_1_anti(rotation_frame_angle, coord, face_1, shape, label_list)  # 37
            path_2 = cal_coord_2(theta, phi, coord, face_2, shape, label_list)  # 16
            numbers = cal_num(coord, path_1, path_2, theta, rotation_frame_angle)  # 3.5s
            absorption = cal_rate(numbers, coefficients, pixel_size)
            absorp[k] = absorption

        print(absorp.mean())

        print('[{}/{}] theta: {:.4f}, phi: {:.4f} , rotation: {:.4f},  absorption: {:.4f}'.format(low + i,
                                                                                low + len(selected_data),
                                                                               theta * 180 / np.pi,
                                                                               phi * 180 / np.pi,
                                                                               rotation_frame_angle* 180 / np.pi,
                                                                                absorp.mean()))
        # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
        corr.append(absorp.mean())
        t2 = time.time()
        print('it spends {}'.format(t2 - t1))

        if args.save_index ==1:
          dict_corr.append({'index': low + i, 'miller_index': miller_index,
                            'intensity': intensity, 'corr': absorp.mean(), 'lp': lp})
          if i%500 ==1:
              with open(os.path.join(save_dir,"{}_refl_{}.json".format(dataset,up)), "w") as fz:  # Pickling
                  json.dump(corr, fz, indent=2)
  
              with open(os.path.join(save_dir,"{}_dict_refl_{}.json".format(dataset,up)), "w") as f1:  # Pickling
                  json.dump(dict_corr, f1, indent=2)

    if args.save_index ==1:
      with open(os.path.join(save_dir, "{}_refl_{}.json".format(dataset, up)), "w") as fz:  # Pickling
          json.dump(corr, fz, indent=2)
  
      with open(os.path.join(save_dir, "{}_dict_refl_{}.json".format(dataset, up)), "w") as f1:  # Pickling
          json.dump(dict_corr, f1, indent=2)

    print('Finish!!!!')








