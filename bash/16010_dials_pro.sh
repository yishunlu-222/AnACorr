#!/bin/sh
experi=16010_best_full_iteration_5_-90_coordset_22_coemy_16010 # dont forget to cp or not overlap the original datafile
save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/${experi}/
sq=sq
#cp -r main3_16010.py  ${save_path}main3_16010_${experi}.py
#cp -r utils.py  ${save_path}utils_${experi}.py

pro_path='/home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/'
python stacking.py --save-dir $save_path --dataset 16010
cp  ${save_path}16010_refl_overall.json   $pro_path
source /home/yishun/packages/dials-v3-6-0/dials
cd $pro_path
python into_flex.py --save-number $experi  --refl-filename  "AUTOMATIC_DEFAULT_SAD_SWEEP1.refl"   #"AUTOMATIC_DEFAULT_SAD_SWEEP1_sorted.refl"
#source /home/yishun/dials/dials-dev20220111/dials_env.sh
source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh
dials.scale test_$experi.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
        model=analytical_absorption anomalous=True  output.reflections=result_${experi}.refl  output.html=result_${experi}_var_${sq}.html  physical.absorption_correction=False \
         output{unmerged_mtz=16010_unmerged_ac_with_dials.mtz}

#source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
#mtz2sca 16010_best_ac.mtz

dials.scale test_$experi.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
       physical.absorption_level=high anomalous=True   model=analytical_absorption   output.reflections=result_${experi}_sh.refl  output.html=result_${experi}_var_${sq}_sh.html \
       output{unmerged_mtz=16010_unmerged_acsh_with_dials.mtz}
#mtz2sca 16010_best_acsh.mtz