#!/bin/sh
kp='km70_pm120'  
experi="best_${kp}_4_-90_(okp)v_nor"    # dont forget to cp or not overlap the original datafile
save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/16010_${experi}/
filename=16010_ompk_10_3p5keV_${kp}_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
expt=16010_ompk_10_3p5keV_${kp}_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt
pro_path='/home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/'
python stacking.py --save-dir $save_path --dataset 16010
cp  ${save_path}16010_refl_overall.json   $pro_path
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
