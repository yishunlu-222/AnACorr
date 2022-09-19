#!/bin/sh
number=14116_m2_inter25_single_scaled_ac_0.019_1_0.0183_0.0382

path=/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/
error_model_a=1.69721
error_model_b=0.03348
error_model_default=True

 # dont forget to cp or not overlap the original datafile
save_path=/home/yishun/projectcode/dials_develop/save_data/from_arc/${number}/
filename=14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl
sq=lin
#cp -r main3_14116.py  ${save_path}main3_14116_${experi}.py
#cp -r utils.py  ${save_path}utils_${experi}.py
pro_path='/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/'
python /home/yishun/projectcode/dials_develop/stacking.py --save-dir $save_path --dataset 14116
cp  ${save_path}14116_refl_overall.json   $pro_path
source /home/yishun/packages/dials-v3-6-0/dials
cd $pro_path
python into_flex.py --save-number $number  --refl-filename $filename --var ${sq}

#single anomalous peak heights
#mtz="14116_merge_ac"
#cd ./anode
#mkdir test_${mtz}
#cp 14116_best_aac.pdb ./test_${mtz}/test_${mtz}.pdb
#cp ../${mtz}.mtz ./test_${mtz}
#cd ./test_${mtz}
#
#source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
#mtz2sca ${mtz}.mtz
#shelxc test_${mtz} << eof
#SAD ${mtz}.sca
#CELL 231.874 74.5031 91.2296 90 112.194 90
#SPAG C121
#eof
#anode test_${mtz}
#sleep 6000  a = 1.69207, b = 0.02300 
#  error_model.basic.a=1.86390 error_model.basic.b=0.02775 
#source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh 
#dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
#physical.absorption_correction=False  anomalous=True output{merged_mtz=test_no.mtz} error_model.basic.a=1.69207 error_model.basic.b=0.02300 # error_model=None   #a = 2.94747, b = 0.01959
#cd ./anode
#mkdir test_no
#cp 14116_refine_sh.pdb ./test_no/test_no.pdb
#cp ../test_no.mtz ./test_no
#cd ./test_no
#source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
#mtz2sca test_no.mtz
#shelxc test_no << eof
#SAD test_no.sca
#CELL 231.874 74.5031 91.2296 90 112.194 90
#SPAG C121
##FIND 6
##MIND -2 -0.1
##SHEL 999 2.0
##SFAC Zn
##NTRY 400
#eof
#anode test_no 
###
###
#cd $path
##source /home/yishun/dials/dials-dev20220111/dials_env.sh
#source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh
#dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl  \
#   physical.absorption_level=high  anomalous=True output{merged_mtz=test_sh.mtz} #error_model=None   #error_model=None\
##error_model.basic.a=${error_model_a}   error_model.basic.b=${error_model_b} \
##error_model.reset_error_model=${error_model_default} \
##output.html=result_a${error_model_a}b${error_model_b}_sh.html   
#
#cd ./anode
#mkdir test_sh
#cp 14116_refine_sh.pdb ./test_sh/test_sh.pdb
#cp ../test_sh.mtz ./test_sh
#cd ./test_sh
#source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
#mtz2sca test_sh.mtz
#shelxc test_sh << eof
#SAD test_sh.sca
#CELL 231.874 74.5031 91.2296 90 112.194 90
#SPAG C121
#FIND 6
#MIND -2 -0.1
#SHEL 999 2.0
#SFAC Zn
#NTRY 400
#eof
#anode test_sh 
####
###sleep 30
##
#cd $path
##source /home/yishun/dials/dials-dev20220111/dials_env.sh   
#source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh
#dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  test_${number}.refl physical.absorption_correction=False anomalous=True  model=analytical_absorption \
#output{merged_mtz=test_${number}.mtz} #output{merged_mtz=14116_merged_ac_with_dials.mtz}  #error_model=None  #output.html=result_a${error_model_a}b${error_model_b}_${number}.html  \
##error_model.basic.a=${error_model_a}   error_model.basic.b=${error_model_b} #error_model=None\
##output.html=result_a${error_model_a}b${error_model_b}_${number}.html 
# #basic.min_Ih=10  
##dials.merge  ../AUTOMATIC_DEFAULT_SAD_SWEEP1.expt ../test_${number}.refl mtz.hklout=test_${number}.mtz
#cd ./anode
#mkdir test_${number}
#cp 14116_refine_aac.pdb ./test_${number}/test_${number}.pdb
#cp ../test_${number}.mtz ./test_${number}
#cd ./test_${number}  
#source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
#mtz2sca test_${number}.mtz
#shelxc test_${number} << eof
#SAD test_${number}.sca
#CELL 231.874 74.5031 91.2296 90 112.194 90
#SPAG C121
#FIND 6
#MIND -2 -0.1
#SHEL 999 2.0
#SFAC Zn
#NTRY 400
#eof
#eof
#anode test_${number} 

#sleep 30

cd $path
#source /home/yishun/dials/dials-dev20220111/dials_env.sh   
source /home/yishun/dials_develop_2/dials-dev20220906/dials_env.sh
dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt test_${number}.refl \
 physical.absorption_level=high anomalous=True model=analytical_absorption  output{merged_mtz=test_${number}_sh.mtz} #error_model=None  \
#output{merged_mtz=14116_merged_acsh_with_dials.mtz} 
#error_model.basic.a=${error_model_a}   error_model.basic.b=${error_model_b}  \
#output.html=result_a${error_model_a}b${error_model_b}_${number}_sh.html   \
#error_model=None   #basic.min_Ih=10 
#dials.merge  ../AUTOMATIC_DEFAULT_SAD_SWEEP1.expt ../test_${number}.refl mtz.hklout=test_${number}.mtz
cd ./anode
mkdir test_${number}_sh
cp 14116_refine_aac.pdb ./test_${number}_sh/test_${number}_sh.pdb
cp ../test_${number}_sh.mtz ./test_${number}_sh
cd ./test_${number}_sh

source /home/yishun/ccp4-7.1/bin/ccp4.setup-sh
mtz2sca test_${number}_sh.mtz
shelxc test_${number}_sh << eof
SAD test_${number}_sh.sca
CELL 231.874 74.5031 91.2296 90 112.194 90
SPAG C121
eof
anode test_${number}_sh 
