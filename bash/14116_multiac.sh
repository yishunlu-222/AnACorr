#!/bin/sh

source /home/yishun/dials/dials-dev20220111/dials_env.sh 
testnumber=0
pro_path='/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/'
start=0
exper=2
bu=1.0
for lo in 0.95 1.0 1.05;
do
  for cr in 0.95 1.0 1.05;
  do
    for li in 0.95 1.0 1.05;
    do
    cd /home/yishun/projectcode/dials_develop/
    testnumber=$[$testnumber+1]
    #expri=${testnumber}_${start}_${li}_${bu}_${cr}
    expri=${testnumber}_${start}_${li}_${lo}_${cr}_${bu}
    echo $expri  
    save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/14116_multiac/new${exper}/14116_test_${expri}/
    python stacking.py --save-dir $save_path --dataset 14116
    cp  ${save_path}14116_refl_overall.json   $pro_path
    cd $pro_path
    dials.python into_flex.py --save-number $expri
    dials.scale test_$expri.refl  AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  anomalous=True  physical.absorption_correction=False
    dials.python dials_merging_stats.py --start ${testnumber}  --end ${start} --save-name "different_ac_13304_multiac_new${exper}.txt" --li ${li} --lo ${lo} --cr ${cr} --bu ${bu}
    
    done
  done
done
