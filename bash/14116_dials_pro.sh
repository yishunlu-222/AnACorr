#!/bin/sh 
experi=14116_m2_v_scaled_ac_0_1_1_1_1
sq=lin
 # dont forget to cp or not overlap the original datafile
save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/${experi}/
filename=14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
#cp -r main3_14116.py  ${save_path}main3_14116_${experi}.py
#cp -r utils.py  ${save_path}utils_${experi}.py
pro_path='/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/'
python stacking.py --save-dir $save_path --dataset 14116
cp  ${save_path}14116_refl_overall.json   $pro_path
source /home/yishun/packages/dials-v3-6-0/dials
cd $pro_path
python into_flex.py --save-number $experi  --refl-filename $filename --var ${sq}
#source /home/yishun/dials/dials-dev20220111/dials_env.sh
source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh
#source /home/yishun/dials_develop_version/dials
dials.scale test_$experi.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
       	 anomalous=True  output.reflections=result_${experi}.refl  output.html=result_${experi}_var_${sq}.html  model=analytical_absorption physical.absorption_correction=False \
          #output{unmerged_mtz=14116_unmerged_ac_with_dials.mtz} # \
#        output.html=result_a1bp02_${experi}.html \
#        error_model.basic.a=1 error_model.basic.b=0.02
         #error_model=None
         

dials.scale test_$experi.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
       physical.absorption_level=high anomalous=True  output.reflections=result_${experi}_sh.refl  output.html=result_${experi}_var_${sq}_sh.html model=analytical_absorption  \
       #output{unmerged_mtz=14116_unmerged_acsh_with_dials.mtz}
#       output.html=result_a1bp02_${experi}_sh.html  \
#       error_model.basic.a=1 error_model.basic.b=0.02 
       #error_model=None
       #output{unmerged_mtz=14116_unmerged_acsh.mtz}
