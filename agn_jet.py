import numpy as np
import proxmin
import deblender
from sys import argv
from scipy.misc import imsave
import fitsio
import csv
import math
import matplotlib.pyplot as plt

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

def plotColorImage(images, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, testing=True):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    import matplotlib.pyplot as plt
    colors = imagesToRgb(images, filterIndices, xRange, yRange, contrast, adjustZero)
    if testing:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.show()
    else:
        	imsave("Test3/%s-A_Data.png" % objName, colors)
    return colors

def plotColorImage2(images, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, testing=True):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    import matplotlib.pyplot as plt
    colors = imagesToRgb(images, filterIndices, xRange, yRange, contrast, adjustZero)
    if testing:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.show()
    else:
        	imsave("Test3/%s-B_Model.png" % objName, colors)
    return colors

def plotComponents(A, S, Tx, Ty, ks=None, filterIndices=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, testing=True):
    import matplotlib.pyplot as plt
    if ks is None:
        ks = range(len(S))
    for k in ks:
        #component = deblender.nmf.get_peak_model(A[:,k], S[k].flatten(), Tx[k], Ty[k], shape=(S[k].shape))[0]
        component = deblender.nmf.get_peak_model(A, S.reshape(len(S),-1), Tx, Ty, shape=(S[k].shape),k=k)
        colors = imagesToRgb(component, filterIndices, xRange, yRange, contrast, adjustZero)
	if testing:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.show()
	else:
        	imsave("Test3/%s-C_Jet.png" % objName, colors)
        #plt.figure()
        #plt.imshow(np.ma.array(component, mask=component==0))
        #plt.show()

# load object-deblending parameters
data = open("objParams.csv","r")
reader = csv.reader(data)
# stores object parameters 0:Name 1:Peaks 2:Peak Detection Multiplier 3:Peak Detection Band Weighting 
objParams = []
for row in reader:
	temp_peaks = np.fromstring(row[1], dtype=int, sep=' ')
	objParams.append([row[0],np.flip(temp_peaks.reshape((len(temp_peaks)/2, 2)),axis=1),int(row[2])])
#print(objParams[14][1])
obj_nums = np.arange(len(objParams))

# set the object number for testing------
obj_nums = [18]

# process objects
for o in obj_nums:
	#try:
	for d in range(1):
		print(objParams[o][0], o)
		# load data and psfs
		path = objParams[o][0]
		data_bands = []
		for b in bands:
		    hdu = fitsio.FITS("%s/stamp-%s.fits" % (path, b))
		    data_bands.append(hdu[0][:,:])
		    hdu.close()
		data = np.array(data_bands)

		psfs = []
		for b in bands:
		    hdu = fitsio.FITS("%s/psf-diff_kernel_%s.fits" % (path, b))
		    psfs.append(hdu[0][:,:])
		    hdu.close()
		psfs = np.array(psfs)
		
		# find weights
		weights = np.ones_like(data)
		band_weights = np.zeros(data.shape[0])
		for i in range(data.shape[0]):
			within = data[i] < 100000
			for t in range(25):
				std = np.std(data[i][within])
				mean = np.mean(data[i][within])
				within = np.bitwise_and(within, (data[i] < mean + 3*std))
				within = np.bitwise_and(within, (data[i] > mean - 3*std))
			band_weights[i] = std
		band_weights = proxmin.operators.prox_unity_plus(band_weights, 1)
		weights *= band_weights[:,None,None]
		#print(band_weights)
		
		# load peak position
		peaks = objParams[o][1]
		constraints = ["m"] * len(peaks)
	
		# add jet component
		shape = data[0].shape
		peaks = np.concatenate((peaks, np.array([shape[0]/2,shape[1]/2])[None,:]),axis=0)
		constraints = constraints + [None]

		# restrict to inner pixels
		inner = 119
		"""
		dx = (shape[0] - inner)/2
		dy = (shape[1] - inner)/2
		data = data[:,dx:-dx,dy:-dy]
		peaks = np.array(peaks) - np.array((dx,dy))
		inside = (peaks[:,0] > 0) & (peaks[:,1] > 0) & (peaks[:,0] < inner) & (peaks[:,1] < inner)
		peaks = peaks[inside]
		constraints = [constraints[i] for i in range(len(constraints)) if inside[i] == 1]
		"""
		# find SEDs
		# read central galaxy and jet colors from spectral data 
		row_num = 0
		with open('%s/spec_mag.csv' % (path), 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				row_num += 1
				if row_num == 6:
					SED_data = row
		jet_sed = np.array([float(SED_data[30]),float(SED_data[31]),float(SED_data[32]),float(SED_data[33]),float(SED_data[34])])
		gal_sed = np.array([float(SED_data[35]),float(SED_data[36]),float(SED_data[37]),float(SED_data[38]),float(SED_data[39])])
		jet_sed = proxmin.operators.prox_unity_plus(jet_sed, 1)
		gal_sed = proxmin.operators.prox_unity_plus(gal_sed, 1)
		
		center_index = 0
		min_dist = 1000
		color_sample_radius = 1
		color_avg_p = 1 # for fancy color means
		SEDs = np.zeros((len(peaks),len(jet_sed)))
		for i in range(len(peaks) - 1):
			# find index of central peak
			curr_dist = np.absolute(peaks[i][0] - inner/2) + np.absolute(peaks[i][1] - inner/2)
			if curr_dist < min_dist:
				min_dist = curr_dist
				center_index = i
			# find observed color of peaks
			count = 0
			for ii in range(-color_sample_radius, color_sample_radius+1):
				for jj in range(-color_sample_radius, color_sample_radius+1):
					try:
						SEDs[i] += data[:,peaks[i][1] + ii, peaks[i][0] + jj]**color_avg_p
						count += 1
					except:
						pass
			SEDs[i] /= count 
			SEDs[i] **= (1/color_avg_p)
			SEDs[i] = proxmin.operators.prox_unity_plus(SEDs[i], 1)
			#SEDs[i] = None
		SEDs[center_index] = gal_sed
		SEDs[-1] = jet_sed
		#print(center_index)
		#print(SEDs)

		# create thresholds
		gal_t = 5e-4
		jet_t = 1.8e-2
		l1_thresh = np.ones(len(peaks))*gal_t
		l1_thresh[-1] = jet_t

		# define constraints
		def prox_SED(A, step, SEDs=None):
		    	for i in range(len(A[0])):
				if not math.isnan(SEDs[i][0]):
					A[:,i] = SEDs[i]
			return proxmin.operators.prox_unity_plus(A, step, axis=0)

		from functools import partial
		prox_A = partial(prox_SED, SEDs=SEDs)
		
		# define masks for localizing jet/galaxies
		radii = np.ones(len(peaks))*500
		radii[-1] = 30
		masks = np.zeros((len(peaks),data.shape[1]*data.shape[2]))
		k = 0.5
		for i in range(masks.shape[0]):
			temp = ((np.arange(data.shape[1]) - peaks[i][0])**2)[:,None] + ((np.arange(data.shape[2]) - peaks[i][1])**2)[None,:]			
			masks[i] = (1/(1 + np.exp(k*(temp**0.5 - radii[i])))).T.ravel()

		def prox_Jet(S, step, l0_thresh=None, l1_thresh=None, masks=None):
			S *= masks			
			if l0_thresh is None and l1_thresh is None:            
				return proxmin.operators.prox_plus(S, step)        
			else:
				# L0 has preference            
				if l0_thresh is not None:                
					if l1_thresh is not None:                    
						return proxmin.operators.prox_hard(S, step, thresh=l0_thresh)
				else:                
					return proxmin.operators.prox_soft_plus(S, step, thresh=l1_thresh)
		
		prox_S = partial(prox_Jet, masks=masks, l1_thresh=l1_thresh[:,None])
		
		# run deblender
		result = deblender.nmf.deblend(data,
		    peaks=peaks, weights=weights,
		    psf=psfs,
		    constraints=constraints,
		    prox_A=prox_A,
		    #prox_gA=None,
		    prox_S=prox_S,
		    monotonicUseNearest=False,
		    max_iter=1000,
		    e_rel=[1e-6,1e-3],
		    #l0_thresh=np.array([5e-3,5e-3,5e-3,5e-1])[:,None],
		    l1_thresh=l1_thresh[:,None],
		    traceback=False,
		    update_order=[1,0])
		A, S, model, P_, Tx, Ty, tr = result
	
		testing = False
		testing = True # comment out to write to files
		contrast = 20
		plotColorImage(data, contrast=contrast, objName=(str(objParams[o][0])[:-1] + "_" + str(o)), testing=testing)
		plotColorImage2(model, contrast=contrast, objName=(str(objParams[o][0])[:-1] + "_" + str(o)), testing=testing)
		plotComponents(A, S, Tx, Ty, ks=[-1], contrast=contrast, objName=(str(objParams[o][0])[:-1] + "_" + str(o)), testing=testing)

"""
	except Exception, e:
		print("FAILED: " + str(e))
		pass
"""

