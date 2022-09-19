#!/bin/sh
#save_path='/home/yishun/projectcode/dials_develop/save_data/16010_test2/'
#pro_path='/home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/'
#python stacking.py --save-dir $save_path --dataset 16010
#cp -r ${save_path}16010_refl_overall.json  $pro_path
#cd $pro_path
source /home/yishun/packages/dials-v3-6-0/dials
experi=38
python into_flex.py --save-number $experi
source /home/yishun/dials/dials-dev20210720/dials_env.sh
dials.scale test_${experi}.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt physical.absorption_correction=False \
anomalous=True
sleep 20
dials.scale test_${experi}.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt absorption_level=high  \
anomalous=True