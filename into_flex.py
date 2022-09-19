import json
import numpy as np
import pdb
import random
import  argparse

parser = argparse.ArgumentParser(description="putting corrected values files into flex files")

parser.add_argument(
    "--save-number",
    type=str,
    default=0,
    help="save-dir for stacking",
)
parser.add_argument(
    "--refl-filename",
    type=str,
    default="",
    help="save-dir for stacking",
)
parser.add_argument(
    "--var",
    type=str,
    default="lin",
    help="save-dir for stacking",
)
global args
args = parser.parse_args()
# pdb.set_trace()



from dials.array_family import flex
refl_filename="14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl"
reflections= flex.reflection_table.from_file(args.refl_filename)

def sum_normalized(raw,after):
    normalized =  np.sum(raw) / np.sum(after) 
    after = after * normalized
    
    return after

corr = np.ones(len(reflections))
p=[]
dataset=14116
filename='{}_refl_overall.json'.format(dataset)

with open(filename) as f1:
  data = json.load(f1)
for i,row in enumerate(data):
    corr[i] =row
    
raw = np.array(reflections['intensity.sum.value'])
after = raw/corr
#after = after / corr.mean()
#after = sum_normalized(raw,after)

raw_prf =np.array(reflections['intensity.prf.value'])
prf_after = raw_prf/corr
#prf_after = after / corr.mean()
#prf_after = sum_normalized(raw_prf,prf_after)

if args.var == 'sq':
  varafter = np.array(reflections['intensity.sum.variance'])/(corr**2)
  prf_varafter = np.array(reflections['intensity.prf.variance'])/(corr**2)
elif args.var == 'lin':
  raw_var = np.array(reflections['intensity.sum.variance'])
  varafter = raw_var/(corr)
  #varafter  = sum_normalized(raw_var,varafter )
  #varafter = varafter /corr.mean()
  
  raw_prf_var =  np.array(reflections['intensity.prf.variance'])
  prf_varafter =  raw_prf_var /(corr)
  #prf_varafter = prf_varafter /corr.mean()
  #prf_varafter   = sum_normalized(raw_prf_var ,prf_varafter )
i_sigma =  after / varafter
i_sigma_prf = prf_after  /prf_varafter

#prf_varafter = np.array(reflections['intensity.prf.variance'])/(corr)
#after = np.array(reflections['intensity.sum.value'])*2
#varafter = np.array(reflections['intensity.sum.variance'])*2
#prf_after = np.array(reflections['intensity.prf.value'])*2
#prf_varafter = np.array(reflections['intensity.prf.variance'])*2

ac = flex.double(list(corr))
reflections["analytical_absorption_correction"] = ac
#afterr=flex.double(list(after))
#varafterr = flex.double(list(varafter))
#prf_afterr=flex.double(list(prf_after))
#prf_varafterr = flex.double(list(prf_varafter))
#reflections['intensity.sum.value'] = afterr
#reflections['intensity.sum.variance'] = varafterr
#reflections['intensity.prf.value'] = prf_afterr
#reflections['intensity.prf.variance'] = prf_varafterr

reflections.as_file("test_{}.refl".format(args.save_number))

