#!/bin/sh
  # dont forget to cp or not overlap the original datafile
kp='km10'
experi=best_${kp}_2_-90_okpv_combor
save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/14116_${experi}/
filename=14116_ompk_19_3keV_${kp}_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
expt=14116_ompk_19_3keV_${kp}_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt
pro_path='/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/'
python stacking.py --save-dir $save_path --dataset 14116
cp  ${save_path}14116_refl_overall.json   $pro_path
source /home/yishun/packages/dials-v3-6-0/dials
cd $pro_path
python into_flex.py --save-number $experi  --refl-filename  $filename    #"AUTOMATIC_DEFAULT_SAD_SWEEP1_sorted.refl"
source /home/yishun/dials/dials-dev20220111/dials_env.sh
dials.scale test_$experi.refl $expt \
       	physical.absorption_correction=False anomalous=True  output.reflections=result_${experi}.refl
#output.reflections="my_${experi}.refl"
sleep 20 
dials.scale test_$experi.refl $expt \
       absorption_level=high anomalous=True output.reflections=result_${experi}_sh.refl
