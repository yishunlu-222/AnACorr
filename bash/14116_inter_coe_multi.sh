#!/bin/sh 

round() {
  printf "%.${2}f" "${1}"
}

li=0.01891746111084849
li_sigma=0.0009918545275298355
cr_25=0.0185079544759556
cr_25_sigma=0.0016409838957426469
bu_25=0.03924892748443271
bu_25_sigma=0.0024716550362188956
li_lower=$(echo "($li-$li_sigma)"| bc -l)
li_upper=$(echo "($li+$li_sigma)"| bc -l)
li_array=($(echo "($li-$li_sigma)"| bc -l)  $li   $(echo "($li+$li_sigma)"| bc -l))
cr_array=( $(echo "($cr_25-$cr_25_sigma)"| bc -l)  $cr_25  $(echo "($cr_25+$cr_25_sigma)"| bc -l) )
bu_array=( $(echo "($bu_25-$bu_25_sigma)"| bc -l)  $bu_25  $(echo "($bu_25+$bu_25_sigma)"| bc -l) )

cr_25=0.0185079544759556
cr_25_sigma=0.0016409838957426469
cr_lower=$(echo "($cr_25-$cr_25_sigma)"| bc -l)
cr_upper=$(echo "($cr_25+$cr_25_sigma)"| bc -l)
#echo $cr_lower
#echo $cr_upper
cr_50=0.01833298951009355
cr_50_sigma=0.001553908884846004
cr_75=0.01891746111084849
cr_75_sigma=0.01891746111084849
cr_100=0.01891746111084849
cr_100_sigma=0.01891746111084849

bu_25=0.03924892748443271
bu_25_sigma=0.0024716550362188956
bu_50=0.04022093282682156
bu_50_sigma=0.0034455811153744397
bu_75=0.01891746111084849
bu_75_sigma=0.01891746111084849
bu_100=0.01891746111084849
bu_100_sigma=0.01891746111084849
filename=14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
pro_path='/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/' 
for liquor in "${li_array[@]}";
do
    for crystal in "${cr_array[@]}";
    do
      for bubble in "${bu_array[@]}";
      do
      
      cd /home/yishun/projectcode/dials_develop/
      #experi=14116_m0_scaled_ac_${i}_1_1_1_1
      liq=$(round ${liquor} 4)
      cry=$(round ${crystal} 4)
      bub=$(round ${bubble} 4)
      experi=14116_m2_inter_scaled_ac_${liq}_1_${cry}_${bub}
      save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/14116_m2_inter/${experi}/
      #save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/14116_scaled_ac/${experi}/
      python stacking.py --save-dir $save_path --dataset 14116
      cp  ${save_path}14116_refl_overall.json   $pro_path
      source /home/yishun/packages/dials-v3-6-0/dials
      cd $pro_path
      python into_flex.py --save-number $experi  --refl-filename $filename --var 'sq'
      source /home/yishun/dials/dials-dev20220111/dials_env.sh
      dials.scale test_$experi.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
           	physical.absorption_correction=False anomalous=True output.reflections=result_${experi}.refl 
          
    	
      dials.scale test_$experi.refl 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt \
           absorption_level=high anomalous=True  output.reflections=result_${experi}_sh.refl  

    done
  done
done


