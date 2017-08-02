import numpy as np
import proxmin
import deblender
from sys import argv
import fitsio

bands = ['g','r','i','z','y']
objname = argv[1]

def imagesToRgb(images, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False):
    """Convert a collection of images or calexp's to an RGB image

    This requires either an array of images or a list of calexps.
    If filter indices is not specified, it uses the first three images in opposite order
    (for example if images=[g, r, i], i->R, r->G, g->B).
    xRange and yRange can be passed to slice an image.
    """
    if len(images)<3:
        raise ValueError("Expected either an array of 3 or more images")
    if filterIndices is None:
        filterIndices = [3,2,1]
    if yRange is None:
        ySlice = slice(None, None)
    elif not isinstance(yRange, slice):
        ySlice = slice(yRange[0], yRange[1])
    else:
        ySlice = yRange
    if xRange is None:
        xSlice = slice(None, None)
    elif not isinstance(xRange, slice):
        xSlice = slice(xRange[0], xRange[1])
    else:
        xSlice = xRange
    # Select the subset of 3 images to use for the RGB image
    images = images[filterIndices,ySlice, xSlice]

    # Map intensity to [0,255]
    intensity = np.arcsinh(contrast*np.sum(images, axis=0)/3)
    if adjustZero:
        # Adjust the colors so that zero is the lowest flux value
        intensity = (intensity-np.min(intensity))/(np.max(intensity)-np.min(intensity))*255
    else:
        maxIntensity = np.max(intensity)
        if maxIntensity > 0:
            intensity = intensity/(maxIntensity)*255
            intensity[intensity<0] = 0

    # Use the absolute value to normalize the pixel intensities
    pixelIntensity = np.sum(np.abs(images), axis=0)
    # Prevent division by zero
    zeroPix = pixelIntensity==0
    pixelIntensity[zeroPix] = 1

    # Calculate the RGB colors
    pixelIntensity = np.broadcast_to(pixelIntensity, (3, pixelIntensity.shape[0], pixelIntensity.shape[1]))
    intensity = np.broadcast_to(intensity, (3, intensity.shape[0], intensity.shape[1]))
    zeroPix = np.broadcast_to(zeroPix, (3, zeroPix.shape[0], zeroPix.shape[1]))
    colors = images/pixelIntensity*intensity
    colors[colors<0] = 0
    colors[zeroPix] = 0
    colors = colors.astype(np.uint8)
    return np.dstack(colors)

def plotColorImage(images, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5)):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    import matplotlib.pyplot as plt
    colors = imagesToRgb(images, filterIndices, xRange, yRange, contrast, adjustZero)
    plt.figure(figsize=figsize)
    plt.imshow(colors)
    plt.show()
    return colors

def plotComponents(A, S, Tx, Ty, ks=None, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5)):
    import matplotlib.pyplot as plt
    if ks is None:
        ks = range(len(S))

    for k in ks:
        #component = deblender.nmf.get_peak_model(A[:,k], S[k].flatten(), Tx[k], Ty[k], shape=(S[k].shape))[0]
        component = deblender.nmf.get_peak_model(A, S.reshape(len(S),-1), Tx, Ty, shape=(S[k].shape),k=k)
        colors = imagesToRgb(component, filterIndices, xRange, yRange, contrast, adjustZero)
        plt.figure(figsize=figsize)
        plt.imshow(colors)
        plt.show()
        #plt.figure()
        #plt.imshow(np.ma.array(component, mask=component==0))
        #plt.show()


# load data and psfs
data_bands = []
for b in bands:
    hdu = fitsio.FITS("%s/stamp-%s.fits" % (objname, b))
    data_bands.append(hdu[0][:,:])
    hdu.close()
data = np.array(data_bands)

psfs = []
for b in bands:
    hdu = fitsio.FITS("%s/psf-diff_kernel_%s.fits" % (objname, b))
    psfs.append(hdu[0][:,:])
    hdu.close()
psfs = np.array(psfs)

# need to get weights
weights = None

# load expected jet SED
#jet_sed = np.zeros(len(bands))
#specdata = np.loadtxt('%s/spec_mag.csv' % (objname))
jet_sed = np.array([0.06598638046801182,0.2032376761774897,1.9325388133884376,0,1.1942058574708945])
jet_sed = proxmin.operators.prox_unity_plus(jet_sed, 1)

# load peak position
peaks = [[60-0.5,59+0.5], [63+0.75,48+0.5], [58,15], [26,38], [55-1,73-1]]
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
    psf=psfs,
    constraints=constraints,
    prox_A=prox_A,
    monotonicUseNearest=False,
    max_iter=1000,
    e_rel=[1e-6,1e-3],
    l0_thresh=np.array([5e-3,5e-3,5e-3,2e-2])[:,None],
    psf_thresh=5e-3,
    traceback=False,
    update_order=[1,0])
A, S, model, P_, Tx, Ty, tr = result

contrast = 10
plotColorImage(data, contrast=contrast)
plotColorImage(model, contrast=contrast)
plotComponents(A, S, Tx, Ty, ks=[0,-1], contrast=contrast)
