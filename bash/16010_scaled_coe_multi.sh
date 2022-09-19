#!/bin/sh 

start=-100
end=100
increment=5
filename=AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
pro_path='/home/yishun/projectcode/dials_develop/16010_ompk_10/dials/DataFiles/'
for i in $(seq $start $increment $end);
do
  cd /home/yishun/projectcode/dials_develop/
  experi=16010_scaled_ac_${i}
  save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/16010_scaled_ac/${experi}/
  python stacking.py --save-dir $save_path --dataset 16010
  cp  ${save_path}16010_refl_overall.json   $pro_path
  source /home/yishun/packages/dials-v3-6-0/dials
  cd $pro_path
  python into_flex.py --save-number $experi  --refl-filename $filename
  source /home/yishun/dials/dials-dev20220111/dials_env.sh
  dials.scale test_$experi.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
         	physical.absorption_correction=False anomalous=True  output.reflections=result_${experi}.refl 
  
  dials.scale test_$experi.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
         absorption_level=high anomalous=True output.reflections=result_${experi}_sh.refl  
done

