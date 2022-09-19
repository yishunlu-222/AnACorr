#!/bin/sh
number=1
#path=/home/yishun/projectcode/dials_develop/14116_ompk_19_3keV_dials/ompk_19_3keV_dials/DataFiles/
source /home/yishun/dials/dials-dev20220111/dials_env.sh 
dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
physical.absorption_correction=False  anomalous=True output{reflections=result_no.refl} output{html=result_no.html}
dials.export  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt result_no.refl mtz.hklout=result_no.mtz
#
dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl \
absorption_level=high   anomalous=True output{reflections=result_sh.refl} output{html=result_sh.html} output{json=14116_sh.json}
dials.export  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt result_sh.refl mtz.hklout=result_sh.mtz
#
dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  test_${number}.refl \
physical.absorption_correction=False  anomalous=True output{reflections=result_${number}.refl} output{html=result_${number}.html} output{json=14116_${number}.json}
dials.export  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt result_${number}.refl mtz.hklout=result_${number}.mtz
#
dials.scale 14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt  test_${number}.refl \
absorption_level=high  anomalous=True output{reflections=result_${number}_sh.refl} output{html=result_${number}_sh.html} output{json=14116_${number}_sh.json}
dials.export  14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt result_${number}_sh.refl mtz.hklout=result_${number}_sh.mtz