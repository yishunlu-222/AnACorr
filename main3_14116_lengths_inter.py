import os
import json
import time
import pdb
import numpy as np
import argparse
from utils import *
from unit_test import *
import re

parser = argparse.ArgumentParser(description="multiprocessing for batches")

parser.add_argument(
    "--dataset",
    type=str,
    default='14116_m2_inter',
    help="the starting point of the batch",
)
parser.add_argument(
    "--pth",
    type=str,
    default='/data/oerc-diamond-crystallography/hert5199/dials_develop/save_data/14116_best_1_-90_coemy_14116_auto_hp5_vp1_112loop/',
    help="dataset number default is 13304",
)

parser.add_argument(
    "--npy",
    type=str,
    required=True,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--proportion",
    type=int,
    default=0,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--li",
    type=float,
    default=1,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--cr",
    type=float,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--bu",
    type=float,
    default=1,
    help="dataset number default is 13304",
)
parser.add_argument(
    "--lo",
    type=float,
    default=1,
    help="dataset number default is 13304",
)
global args
args = parser.parse_args()

def sort_key(s):

    if s:
        try:
            c = re.findall('(\d+)', s)[1]
        except:
            c = -1
        return int(c)

if __name__ == "__main__":
    # dir = '/data/oerc-diamond-crystallography/hert5199/dials_develop/save_data/14116_best_1_-90_coemy_14116_auto_hp5_vp1_112loop/'
    dataset = args.dataset
    save_dir = './save_data/{}_scaled_ac_{}_{}_{}_{}'.format( dataset,np.round(args.li,4),args.lo,np.round(args.cr,4),np.round(args.bu,4) )
    try:
      if os.path.exists( save_dir ) is False :
          os.mkdir( save_dir )
    except:
      pass
    #propt = args.proportion /100
    npy_list = []
    mu_cr = args.cr*1e3 # (unit in mm-1) 16010
    mu_li = args.li*1e3
    mu_lo = 0.0162e3 
    mu_bu = args.bu*1e3
#    mu_cr = 0.0192e3 * (1 + propt*args.cr) # (unit in mm-1) 16010
#    mu_li = 0.0184e3 * (1 + propt*args.li)
#    mu_lo = 0.0162e3 * (1 + propt*args.lo)
#    mu_bu = 0.046e3 * (1 + propt*args.bu)
    print(mu_cr)
    print(mu_li)
    print(mu_lo)
    print(mu_bu)
    pixel_size = 0.3e-3
    corr = []

    t1 =time.time()
    coefficients = mu_li, mu_lo, mu_cr, mu_bu
    data = np.load(args.npy)
    index =args.npy.split('.')[0]
    index = index.split('_')[-1]
    for refl in data :
        absorp = np.empty( len( refl ) )
        for k, pixel in enumerate(refl):
            numbers = pixel
            absorption = cal_rate( numbers , coefficients , pixel_size )
            absorp[k] = absorption
        corr.append( absorp.mean( ) )


    with open(os.path.join(save_dir, "{}_refl_scaledac_{}.json".format(dataset,index)), "w") as fz:  # Pickling
        json.dump(corr, fz, indent=2)

    # for file in os.listdir( dir ) :
    #
    #     if '.npy' in file:
    #         if '-1' in file :
    #             npy_last = file
    #             continue
    #         npy_list.append( file )
    # npy_list.sort( key = sort_key )
    # if npy_last :
    #     npy_list.append( npy_last )
    #
    # for i, npy in enumerate(npy_list):
    #     data = np.load(os.path.join(dir,npy))
    #     print( 'calculating on {}'.format( npy ) )
    #     print( '{}/{}'.format(i,len(npy_list)))
    #     for refl in data :
    #         absorp = np.empty( len( refl ) )
    #         t2 = time.time()
    #         for k, pixel in enumerate(refl):
    #             numbers = pixel
    #             absorption = cal_rate( numbers , coefficients , pixel_size )
    #             absorp[k] = absorption
    #         corr.append( absorp.mean( ) )
    #         print( 'one ray needs {}'.format( time.time( ) - t2 ) )
    #     print('finish one npy needs {}'.format(time.time()-t1))
    #     t1 =time.time()
    # with open(os.path.join(save_dir, "{}_refl_scaledac_overall.json".format(dataset)), "w") as fz:  # Pickling
    #     json.dump(corr, fz, indent=2)
