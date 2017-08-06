import numpy as np
import os, fitsio
from sys import argv
import deblender
from scipy.misc import imsave
import matplotlib.pyplot as plt
import proxmin

def displayResults(data,result,contrast=20,filterWeights=None,writeFile=False,objName="NONAME",o=0, folderName=None, extra_center=False, ks=[-1]):
	# display results	
	A, S, model, P_, Tx, Ty, tr = result
	plotColorImage(data, contrast=contrast, objName=(objName + "_" + str(o) + "-A_Data"), filterWeights=filterWeights, writeFile=writeFile, folderName=folderName)

	plotColorImage(model, contrast=contrast, filterWeights=filterWeights, objName=(objName + "_" + str(o) + "-B_Model"), writeFile=writeFile, folderName=folderName)

	# model central galaxy
	model = np.zeros_like(data)
	for i in range(2):
		model += A[:,i,None,None]*S[None,i,:,:]
	if extra_center:
		plotColorImage(model, contrast=contrast, filterWeights=filterWeights, objName=(objName + "_" + str(o) + "-C_Galaxy"), writeFile=writeFile, folderName=folderName)

	plotComponents(A, S, Tx, Ty, ks=[0], contrast=contrast, filterWeights=filterWeights, objName=(objName + "_" + str(o) + "-D_Main"), writeFile=writeFile, folderName=folderName)

	if extra_center:
		plotComponents(A, S, Tx, Ty, ks=[1], contrast=contrast, filterWeights=filterWeights, objName=(objName + "_" + str(o) + "-E_Peak"), writeFile=writeFile, folderName=folderName)

	plotComponents(A, S, Tx, Ty, ks=ks, contrast=contrast, filterWeights=filterWeights, objName=(objName + "_" + str(o) + "-F_Jet"), writeFile=writeFile, folderName=folderName)
	if not writeFile:
		plt.show()

def imagesToRgb(images, filterWeights=None, xRange=None, yRange=None, contrast=1, adjustZero=False):
    """Convert a collection of images or calexp's to an RGB image

    This requires either an array of images or a list of calexps.
    If filter indices is not specified, it uses the first three images in opposite order
    (for example if images=[g, r, i], i->R, r->G, g->B).
    xRange and yRange can be passed to slice an image.
    """
    B = len(images)
    channels = ['R','G','B']

    if filterWeights is None:
        filterWeights = np.array([np.zeros(B) for c in channels])
        filterWeights[0,3] = 1 # R: 100% z
        filterWeights[1,2] = 1 # G: 100% i
        filterWeights[2,1] = 1 # B: 100% r
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
    images = images[:,ySlice, xSlice]
    _,ny,nx = images.shape
    images = np.dot(filterWeights,images.reshape(B,-1)).reshape(-1,ny,nx)

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

def plotColorImage(images, filterWeights=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, writeFile=False, folderName=None):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    colors = imagesToRgb(images, filterWeights, xRange, yRange, contrast, adjustZero)
    if not writeFile:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.title(objName)
    else:
        	imsave("%s/%s.png" % (folderName, objName), colors)
    return colors

def plotComponents(A, S, Tx, Ty, ks=None, filterWeights=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, writeFile=False, folderName=None):
    if ks is None:
        ks = range(len(S))
    for k in ks:
        #component = deblender.nmf.get_peak_model(A[:,k], S[k].flatten(), Tx[k], Ty[k], shape=(S[k].shape))[0]
        component = deblender.nmf.get_peak_model(A, S.reshape(len(S),-1), Tx, Ty, shape=(S[k].shape),k=k)
        colors = imagesToRgb(component, filterWeights, xRange, yRange, contrast, adjustZero)
	if not writeFile:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.title(objName)
	else:
        	imsave("%s/%s.png" % (folderName, objName), colors)
