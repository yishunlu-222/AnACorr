from dials.algorithms.scaling.scaling_library import \
    merging_stats_from_scaled_array,scaled_data_as_miller_array
from dials.array_family import flex
from dxtbx.serialize import load
import pdb
import  argparse
import json
import os
parser = argparse.ArgumentParser(description="putting corrected values files into flex files")

parser.add_argument(
    "--start",
    type=str,
    default='AUTOMATIC_DEFAULT_SAD_SWEEP1.expt',
    help="save-dir for stacking",
)
parser.add_argument(
    "--end",
    type=str,
    default='raw_ordering.refl',
    help="save-dir for stacking",
)
parser.add_argument(
    "--save-name",
    type=str,
    default="differernt_angles.txt",
    help="save-dir for stacking",
)
parser.add_argument(
    "--li",
    type=str,
    default="1",
    help="save-dir for stacking",
)
parser.add_argument(
    "--lo",
    type=str,
    default="1",
    help="save-dir for stacking",
)
parser.add_argument(
    "--cr",
    type=str,
    default="1",
    help="save-dir for stacking",
)
parser.add_argument(
    "--bu",
    type=str,
    default="1",
    help="save-dir for stacking",
)
global args
args = parser.parse_args()
refls = [flex.reflection_table.from_file("scaled.refl")]
expt = load.experiment_list("AUTOMATIC_DEFAULT_SAD_SWEEP1.expt", check_format=False)[0]
experiments=[expt,expt]
scaled_miller_array = scaled_data_as_miller_array(refls,experiments)
stats= merging_stats_from_scaled_array(scaled_miller_array)
stats_dict=stats[0].as_dict()
r_merge=stats_dict['overall']['r_merge']
cc_anom=stats_dict['overall']['cc_anom']
n_obs=stats_dict['overall']['n_obs']
i_over_sigma_mean=stats_dict['overall']['i_over_sigma_mean']
writing = open(args.save_name, "a")
writing.write('{}, li: {}, lo: {}, cr: {},bu :{}, r_merge: {}, cc_anom: {}, n_obs: {}, i_over_sigma_mean: {},  \n'.format(args.start, args.li,args.lo,args.cr,args.bu, r_merge, cc_anom, n_obs, i_over_sigma_mean ))
writing.close()
