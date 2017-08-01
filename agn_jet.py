import numpy as np
import proxmin
import deblender
from sys import argv

objname = argv[1]

# load data
#import astropy
import fitsio
bands = ['g','r','i','z','y']
data_bands = []
for b in bands:
    hdu = fitsio.FITS("%s/stamp-%s.fits" % (objname, b))
    data_bands.append(hdu[0][:,:])
    hdu.close()
data = np.array(data_bands)

# need to get weights
weights = None

# load expected jet SED
#jet_sed = np.zeros(len(bands))
#specdata = np.loadtxt('%s/spec_mag.csv' % (objname))
jet_sed = np.array([0.06598638046801182,0.2032376761774897,1.9325388133884376,0,1.1942058574708945])
#jet_sed = proxmin.operators.prox_unity_plus(jet_sed, 1)

# load peak position
peaks = [[60,59], [63,48], [58,15], [26,38], [55,73]]
constraints = ["mS"] * len(peaks)

# add jet component
peaks = peaks + [peaks[0]]
constraints = constraints + [None]

# restrict to inner 49 pixels
shape = data[0].shape
dx = (shape[0] - 49)/2
dy = (shape[1] - 49)/2
data = data[:,dx:-dx,dy:-dy]
peaks = np.array(peaks) - np.array((dx,dy))
inside = (peaks[:,0] > 0) & (peaks[:,1] > 0)
peaks = peaks[inside]
constraints = [constraints[i] for i in range(len(constraints)) if inside[i] == 1]

# define constraints
def prox_SED(A, step, jet_sed=None):
    A[:,-1] = jet_sed
    return proxmin.operators.prox_unity_plus(A, step, axis=0)

from functools import partial
prox_A = partial(prox_SED, jet_sed=jet_sed)

# run deblender
result = deblender.nmf.deblend(data,
    peaks=peaks, weights=weights,
    psf=None,
    constraints=constraints,
    prox_A=prox_A,
    monotonicUseNearest=False,
    max_iter=10,
    e_rel=[1e-6,1e-3],
    l0_thresh=5e-4,
    traceback=False,
    update_order=[1,0])
A, S, model, P_, Tx, Ty, tr = result

"""
for k in range(len(S)):
    component = deblender.nmf.get_peak_model(A[:,k], S[k].flatten(), Tx[k], Ty[k], shape=(S[k].shape))[0]
    figure()
    imshow(np.ma.array(component, mask=component==0))
"""
