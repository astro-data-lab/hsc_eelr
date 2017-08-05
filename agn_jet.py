import numpy as np
import proxmin
import deblender
from sys import argv
from scipy.misc import imsave
import fitsio
import csv
import math
import matplotlib.pyplot as plt
from plotting import imagesToRgb

bands = ['g','r','i','z','y']

def plotColorImage(images, filterWeights=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, testing=True):
    """Display a collection of images or calexp's as an RGB image

    See `imagesToRgb` for more info.
    """
    import matplotlib.pyplot as plt
    colors = imagesToRgb(images, filterWeights, xRange, yRange, contrast, adjustZero)
    if testing:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.title(objName)
    else:
        	imsave("Test6/%s.png" % objName, colors)
    return colors

def plotComponents(A, S, Tx, Ty, ks=None, filterWeights=None, xRange=None, yRange=None, contrast=1, adjustZero=False, figsize=(5,5), objName=None, testing=True):
    import matplotlib.pyplot as plt
    if ks is None:
        ks = range(len(S))
    for k in ks:
        #component = deblender.nmf.get_peak_model(A[:,k], S[k].flatten(), Tx[k], Ty[k], shape=(S[k].shape))[0]
        component = deblender.nmf.get_peak_model(A, S.reshape(len(S),-1), Tx, Ty, shape=(S[k].shape),k=k)
        colors = imagesToRgb(component, filterWeights, xRange, yRange, contrast, adjustZero)
	if testing:
		plt.figure(figsize=figsize)
		plt.imshow(colors)
		plt.title(objName)
	else:
        	imsave("Test6/%s.png" % objName, colors)
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
obj_nums = [8]
extra_center = False #Enable color gradient correction for the central galaxy
extra_center = True

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
		

		# reshape if necessary
		inner = 59
		s_index = 1
		if data.shape[1] < 119:
			s_index = 2
		if data.shape[2] < 119:
			s_index = 1
		
		# load peak positions
		peaks = objParams[o][1]
		constraints = ["m"] * len(peaks)
	
		# add extra component for center galaxy (Inner galaxy=0, Outer galaxy=1)
		center_index = 0
		min_dist = 1000		
		for i in range(len(peaks)):
			# find index of central peak
			curr_dist = np.absolute(peaks[i][0] - 59) + np.absolute(peaks[i][1] - 59)
			if curr_dist < min_dist:
				min_dist = curr_dist
				center_index = i
		# interchange center peak with first index
		temp = peaks[0].copy()
		peaks[0] = peaks[center_index]
		peaks[center_index] = temp
		temp = constraints[0]
		constraints[0] = constraints[center_index]
		constraints[center_index] = temp
		if extra_center:
			peaks = np.concatenate((np.array(peaks[0])[None,:], peaks),axis=0)
			constraints = [constraints[0]] + constraints

		# add jet component (Jet=-1)
		shape = data[0].shape
		peaks = np.concatenate((peaks, np.array([shape[0]/2,shape[1]/2])[None,:]),axis=0)
		constraints = constraints + [None]

		# restrict to inner pixels
		if inner < 119:
			dx = (shape[0] - inner)/2
			dy = (shape[1] - inner)/2
			data = data[:,dx:-dx,dy:-dy]
			peaks = np.array(peaks) - np.array((dx,dy))
			inside = (peaks[:,0] > 0) & (peaks[:,1] > 0) & (peaks[:,0] < inner) & (peaks[:,1] < inner)
			peaks = peaks[inside]
			constraints = [constraints[i] for i in range(len(constraints)) if inside[i] == 1]
		
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
	
		# find SEDs
		# read central galaxy and jet colors from spectral data
		row_num = 0
		with open('%s/spec_mag.csv' % (path), 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				row_num += 1
				if row_num == 6:
					SED_data = row
		jet_sed = np.array([ 0.0194308, 0.05984675,  0.5690685,   0.,          0.35165396])
#np.array([float(SED_data[30]),float(SED_data[31]),float(SED_data[32]),float(SED_data[33]),float(SED_data[34])])
		gal_sed = np.array([float(SED_data[35]),float(SED_data[36]),float(SED_data[37]),float(SED_data[38]),float(SED_data[39])])
		jet_sed = proxmin.operators.prox_unity_plus(jet_sed, 1)
		gal_sed = proxmin.operators.prox_unity_plus(gal_sed, 1)

		color_sample_radius = 1
		color_avg_p = 1 # for fancy color means
		SEDs = np.zeros((len(peaks),len(jet_sed)))
		for i in range(1,len(peaks) - 1):
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
		SEDs[0] = gal_sed
		SEDs[-1] = jet_sed
		#print(center_index)
		#print(SEDs)

		# create thresholds
		gal_t = 5e-4
		jet_t = 1e-2
		l1_thresh = np.ones(len(peaks))*gal_t
		l1_thresh[-1] = jet_t
		
		# define constraints
		fiber_mask = np.zeros(data.shape[s_index]**2)
		radius = 6
		k = 1 # apodization cutoff slope
		temp = ((np.arange(data.shape[s_index]) - data.shape[s_index]/2)**2)[:,None] + ((np.arange(data.shape[s_index]) - data.shape[s_index]/2)**2)[None,:]
		fiber_mask = (1/(1 + np.exp(k*(temp**0.5 - radius)))).T.ravel()
		fiber_mask = fiber_mask/fiber_mask.sum()
		
		def prox_SED(A, step, SEDs=None, Xs=None, extra_center=False, it=10):
			if extra_center and (step < 0.0005):
				S = Xs[1]
				model = np.dot(A[:,0:2], S[0:2,:])
				model *= fiber_mask[None,:]
				fiber_sum = proxmin.operators.prox_unity_plus(model.sum(axis=1), 1)
				A -= (fiber_sum - SEDs[0])[:,None]
				#print(fiber_sum)
			    	for i in range(2, len(A[0])):
					if not math.isnan(SEDs[i][0]):
						A[:,i] = SEDs[i]	
			else:
				#print("SET")
				for i in range(1,len(A[0])):
					if not math.isnan(SEDs[i][0]):
						A[:,i] = SEDs[i]
			return proxmin.operators.prox_unity_plus(A, step, axis=0)

		from functools import partial
		prox_A = partial(prox_SED, SEDs=SEDs, extra_center=extra_center)

		# define masks for localizing jet/galaxies
		radii = np.ones(len(peaks))*50
		if extra_center:
			radii[1] = 20
		radii[-1] = 50
		masks = np.zeros((len(peaks),data.shape[s_index]**2))
		k = 0.5 # apodization cutoff slope
		for i in range(masks.shape[0]):
			temp = ((np.arange(data.shape[s_index]) - data.shape[s_index]/2)**2)[:,None] + ((np.arange(data.shape[s_index]) - data.shape[s_index]/2)**2)[None,:]
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

		#print(peaks)
		#print(SEDs)
		# run deblender
		result = deblender.nmf.deblend(data,
		    peaks=peaks, weights=weights,
		    psf=psfs,
		    constraints=constraints,
		    prox_A=prox_A,
		    #prox_gA=prox_gA,
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

        	filterWeights = np.zeros((3, len(bands)))
        	filterWeights[0,4] = 1
        	filterWeights[0,3] = 0.666
        	filterWeights[1,3] = 0.333
        	filterWeights[1,2] = 1
        	filterWeights[1,1] = 0.333
        	filterWeights[2,1] = 0.666
        	filterWeights[2,0] = 1
		plotColorImage(data, contrast=contrast, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-A_Data"), filterWeights=filterWeights, testing=testing)
		plotColorImage(model, contrast=contrast, filterWeights=filterWeights, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-B_Model"), testing=testing)
		plotComponents(A, S, Tx, Ty, ks=[-1], contrast=contrast, filterWeights=filterWeights, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-F_Jet"), testing=testing)
		plotComponents(A, S, Tx, Ty, ks=[0], contrast=contrast, filterWeights=filterWeights, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-D_Main"), testing=testing)
		plotComponents(A, S, Tx, Ty, ks=[1], contrast=contrast, filterWeights=filterWeights, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-E_Peak"), testing=testing)
		# model central galaxy
		model = np.zeros_like(data)
		for i in range(2):
			model += A[:,i,None,None]*S[None,i,:,:]
		plotColorImage(model, contrast=contrast, filterWeights=filterWeights, objName=(str(objParams[o][0])[:-1] + "_" + str(o) + "-C_Galaxy"), testing=testing)
		plt.show()
"""
	except Exception, e:
		print("FAILED: " + str(e))
		pass
"""
