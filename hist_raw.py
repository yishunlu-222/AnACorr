import matplotlib.pyplot as plt
import numpy as np
import pdb
from dials.array_family import flex
import json

refl_filename="14116_ompk_19_3keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl"
reflections= flex.reflection_table.from_file(refl_filename)
number_bins=200
inten = np.array(reflections['intensity.sum.value'])

inten_sigma=np.array(reflections['intensity.sum.variance'])

prf= np.array(reflections['intensity.prf.value'])

prf_sigma= np.array(reflections['intensity.prf.variance'])
i_sigma=inten / inten_sigma
prfi_sigma=prf / prf_sigma

inten[inten<0]=1
inten_sigma[inten_sigma<0]=-1
prf[prf<0]=-1
prf_sigma[prf_sigma<0]=-1
inten[inten>1e4]=-1
inten_sigma[inten_sigma>2.5e5]=-1
prf[prf>1e4]=-1
prf_sigma[prf_sigma>2.5e5]=-1


corr = np.ones(len(reflections))
filename='14116_refl_overall_m2_scaled_ac_0_1_1_1_1.json'

with open(filename) as f1:
  data = json.load(f1)
for i,row in enumerate(data):
    corr[i] = 1/row
after = np.array(reflections['intensity.sum.value'])*corr
varafter = np.array(reflections['intensity.sum.variance'])*np.square(corr)
prf_after = np.array(reflections['intensity.prf.value'])*corr
prf_varafter = np.array(reflections['intensity.prf.variance'])*np.square(corr)
i_sigma_after=after / varafter
prf_i_sigma_after =prf_after / prf_varafter

after[after<0]=-1
varafter [varafter <0]=-1
prf_after [prf_after <0]=-1
prf_varafter[prf_varafter<0]=-1
after[after>1e4]=-1
varafter [varafter >2.5e5]=-1
prf_after [prf_after >1e4]=-1
prf_varafter[prf_varafter>2.5e5]=-1

fig, ax1 = plt.subplots(nrows=2, ncols=2)
ax1[0][0].hist( inten , number_bins , facecolor = 'g' )
ax1[0][0].set_title('intensities')
ax1[1][0].hist( inten_sigma , number_bins , facecolor = 'g' )
ax1[1][0].set_title('inten_sigma')
ax1[0][1].hist( after , number_bins , facecolor = 'g' )
ax1[0][1].set_title('intensities / AF')
ax1[1][1].hist( varafter , number_bins , facecolor = 'g' )
ax1[1][1].set_title('inten_sigma / AF^2')
plt.show()
fig, ax2 = plt.subplots(nrows=2, ncols=2)
ax2[0][0].hist( prf , number_bins , facecolor = 'g' )
ax2[0][0].set_title('prf_intensities')
ax2[1][0].hist( prf_sigma , number_bins , facecolor = 'g' )
ax2[1][0].set_title('prf_intensities sigma')
ax2[0][1].hist( prf_after, number_bins , facecolor = 'g' )
ax2[0][1].set_title('prf_intensities after / AF')
ax2[1][1].hist( prf_varafter, number_bins , facecolor = 'g' )
ax2[1][1].set_title('prf_intensities sigma after / AF^2')
plt.show()
fig, ax3 = plt.subplots(nrows=2, ncols=2)
ax3[0][0].hist( i_sigma, number_bins , facecolor = 'g' )
ax3[0][0].set_title('i/(sigma^2)')
ax3[1][0].hist( i_sigma_after , number_bins , facecolor = 'g' )
ax3[1][0].set_title('i/(sigma^2) after correct')
ax3[0][1].hist( prfi_sigma, number_bins , facecolor = 'g' )
ax3[0][1].set_title('prf i/(sigma^2)')
ax3[1][1].hist( prf_i_sigma_after, number_bins , facecolor = 'g' )
ax3[1][1].set_title('prf i/(sigma^2) after correct')
plt.show()
pdb.set_trace()