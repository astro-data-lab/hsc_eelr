import numpy as np
import proxmin
import deblender
from sys import argv
import fitsio
import csv
import math
import matplotlib.pyplot as plt
from plotting import displayResults, plotColorImage
from functools import partial
import traceback

bands = ['g','r','i','z','y']

def loadData(objParams, dimension=119, galaxy_constraint = "M", bleed=None):
	# load data and psfs
	path = objParams[0]
	data_bands = []
	for b in bands:
	    hdu = fitsio.FITS("%s/stamp-%s.fits" % (path, b))
	    data_bands.append(hdu[0][:,:])
	    hdu.close()
	data = np.array(data_bands)

	# load peak positions
	peaks = objParams[1]
	constraints = [galaxy_constraint] * len(peaks)

	# restrict to inner pixels
	dx = (data.shape[1] - dimension)/2
	dy = (data.shape[1] - dimension)/2
	data = data[:,dx:-dx,dy:-dy]
	# reshape if necessary
	new_dim = np.minimum(data.shape[1], data.shape[2])
	new_dim = new_dim - ((new_dim + 1) % 2)
	data = data[:,:new_dim,:new_dim]
	peaks = np.array(peaks) - np.array((dx,dy))
	inside = (peaks[:,0] > 0) & (peaks[:,1] > 0) & (peaks[:,0] < data.shape[1]) & (peaks[:,1] < data.shape[1])
	peaks = peaks[inside]
	constraints = [constraints[i] for i in range(len(constraints)) if inside[i] == 1]
	
	psfs = []
	for b in bands:
	    hdu = fitsio.FITS("%s/psf-diff_kernel_%s.fits" % (path, b))
	    psfs.append(hdu[0][:,:])
	    hdu.close()
	psfs = np.array(psfs)

	# find SEDs
	# read central galaxy and jet colors from spectral data
	row_num = 0
	with open('%s/spec_mag.csv' % (path), 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			row_num += 1
			if row_num == 6:
				SED_data = row
	if bleed is None:
		bleed = np.array([0,0,0,0,0])
	gal_sed = np.array([float(SED_data[25]),float(SED_data[26]),float(SED_data[27]),float(SED_data[28]),float(SED_data[29])]) + bleed
	jet_sed = np.array([float(SED_data[30]),float(SED_data[31]),float(SED_data[32]),float(SED_data[33]),float(SED_data[34])]) - bleed
	print(jet_sed)
	#gal_sed = np.array([float(SED_data[35]),float(SED_data[36]),float(SED_data[37]),float(SED_data[38]),float(SED_data[39])])
	
	# manually overriding found y-band value (seems to often be corrupted by galaxy lines)
	#jet_sed = np.array([0.08,0.17,0.55,0,0])#colorSample(data,32,9, color_sample_radius=0)#np.array([0.08094098, 0.11424346, 0.48816309, 0.1028126, 0.21383987])
	jet_sed = proxmin.operators.prox_unity_plus(jet_sed, 1)
	gal_sed = proxmin.operators.prox_unity_plus(gal_sed, 1)
	return data, psfs, peaks, constraints, jet_sed, gal_sed

def processData(peaks, constraints, data, extra_center=False):
	# add extra component for center galaxy (Inner galaxy=0, Outer galaxy=1)
	center_index = 0
	min_dist = 1000	
	for i in range(len(peaks)):
		# find index of central peak
		curr_dist = np.absolute(peaks[i][0] - data.shape[1]/2) + np.absolute(peaks[i][1] - data.shape[1]/2)
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
	peaks = np.concatenate((peaks, np.array([data.shape[1]/2,data.shape[1]/2])[None,:]),axis=0)
	constraints = constraints + [None]
	return peaks, constraints

def defineParameters(data, peaks, jet_sed, gal_sed, color_sample_radius=1, color_avg_p=1, gal_t=5e-4, jet_t=1e-2, color_others=True):
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
	
	SEDs = np.zeros((len(peaks),len(bands)))
	for i in range(1,len(peaks) - 1):
		SEDs[i] = colorSample(data, peaks[i][1], peaks[i][0],color_sample_radius=color_sample_radius,color_avg_p=color_avg_p)
		if not color_others:
			SEDs[i] = None
	SEDs[0] = gal_sed
	SEDs[-1] = jet_sed
	
	# create thresholds
	l1_thresh = np.ones(len(peaks))*gal_t
	l1_thresh[-1] = jet_t
	return weights, SEDs, l1_thresh

def defineConstraints(data, peaks, SEDs, extra_center=False, l1_thresh=None, fiber_radius=6, fiber_slope=1, galaxy_radius=50, extra_radius=20, jet_radius=40, general_slope=0.5):
	
	# define constraints
	fiber_mask = np.zeros(data.shape[1]**2)
	temp = ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[:,None] + ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[None,:]
	fiber_mask = (1/(1 + np.exp(fiber_slope*(temp**0.5 - fiber_radius)))).T.ravel()
	fiber_mask = fiber_mask/fiber_mask.sum()
		
	def prox_SED(A, step, SEDs=None, Xs=None, extra_center=False, use_absolute=False):
		if extra_center and (step < 0.0005):
			S = Xs[1]
			model = np.dot(A[:,0:2], S[0:2,:])
			model *= fiber_mask[None,:]
			fiber_sum = proxmin.operators.prox_unity_plus(model.sum(axis=1), 1)
			A[:,0:2] -= (fiber_sum - SEDs[0])[:,None]
		    	for i in range(2, len(A[0])):
				if not math.isnan(SEDs[i][0]):
					A[:,i] = SEDs[i]	
			if use_absolute:
				model = np.dot(A[:,0:2], S[0:2,:])
				model *= fiber_mask[None,:]
				fiber_sum = model.sum(axis=1)
		
				model = np.dot(A[:,-1:], S[-1:,:])
				model *= fiber_mask[None,:]
				fiber_sum = model.sum(axis=1)
		else:
			for i in range(0,len(A[0])):
				if not math.isnan(SEDs[i][0]):
					A[:,i] = SEDs[i]
		

		return proxmin.operators.prox_unity_plus(A, step, axis=0)
	prox_A = partial(prox_SED, SEDs=SEDs, extra_center=extra_center)

	# define masks for localizing jet/galaxies
	radii = np.ones(len(peaks))*galaxy_radius
	if extra_center:
		radii[1] = extra_radius
	radii[-1] = jet_radius
	radii[0] = 40
	masks = np.zeros((len(peaks),data.shape[1]**2))
	for i in range(masks.shape[0]):
		temp = ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[:,None] + ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[None,:]
		masks[i] = (1/(1 + np.exp(general_slope*(temp**0.5 - radii[i])))).T.ravel()

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
	return prox_A, prox_S

def colorSample(data, x, y, color_sample_radius=1, color_avg_p=1):
	count = 0
	color = np.zeros(len(bands))
	for ii in range(-color_sample_radius, color_sample_radius+1):
		for jj in range(-color_sample_radius, color_sample_radius+1):
			try:
				color += data[:,x + ii, y + jj]**color_avg_p
				count += 1
			except:
				pass
	color /= count
	color **= (1/color_avg_p)
	color = proxmin.operators.prox_unity_plus(color, 1)
	return color

def deblend(objParams, dimension=119, extra_center=False, color_sample_radius=1, color_avg_p=1, gal_t=5e-4, jet_t=1e-2, fiber_radius=6, fiber_slope=1, galaxy_radius=50, extra_radius=20, jet_radius=40, general_slope=0.5, color_others=True, galaxy_constraint="M", bleed=None, monotonicUseNearest=False, max_iter=1000, e_rel=[1e-6,1e-3], traceback=False, update_order=[1,0]):
	
	data, psfs, peaks, constraints, jet_sed, gal_sed = loadData(objParams, dimension=dimension, galaxy_constraint=galaxy_constraint, bleed=bleed)

	peaks, constraints = processData(peaks, constraints, data, extra_center=extra_center)

	weights, SEDs, l1_thresh = defineParameters(data, peaks, jet_sed, gal_sed, color_sample_radius=color_sample_radius, color_avg_p=color_avg_p, gal_t=gal_t, jet_t=jet_t, color_others=color_others)
	
	prox_A, prox_S = defineConstraints(data, peaks, SEDs, extra_center=extra_center, l1_thresh=l1_thresh, fiber_radius=fiber_radius, fiber_slope=fiber_slope, galaxy_radius=galaxy_radius, extra_radius=extra_radius, jet_radius=jet_radius, general_slope=general_slope)

	print(SEDs)

	# run deblender
	result = deblender.nmf.deblend(data,
	    peaks=peaks, weights=weights,
	    psf=psfs,
	    constraints=constraints,
	    prox_A=prox_A,
	    prox_S=prox_S,
	    monotonicUseNearest=monotonicUseNearest,
	    max_iter=max_iter,
	    e_rel=e_rel,
	    l1_thresh=l1_thresh[:,None],
	    traceback=traceback,
	    update_order=update_order)
	A, S, model, P_, Tx, Ty, tr = result
	
	return result, data

# load stored object-deblending parameters
read_data = open("objParams.csv","r")
reader = csv.reader(read_data)
objParams = []
for row in reader:
	temp_peaks = np.fromstring(row[1], dtype=int, sep=' ')
	objParams.append([row[0],np.flip(temp_peaks.reshape((len(temp_peaks)/2, 2)),axis=1),int(row[2])])

# set the object number for testing------
full = False
if full:
	obj_nums = [35]#np.arange(len(objParams))
else:
	obj_nums = [1]

# process objects
for object_index in obj_nums:
	try:
		print(objParams[object_index][0], object_index)
		extra_center=True
		result, data = deblend(objParams[object_index], 
			extra_center=extra_center, 
			max_iter=1000,
			galaxy_radius=15,
			general_slope=1,
			jet_radius=50,
			galaxy_constraint="M",
			bleed=np.array([0,0,0,0,0]),
			dimension=85)
		"""
		objParams,			objParams[0]: Object Name objParams[1]: Nx2 numpy array of image peaks
		dimension=119, 			Final Image will be dimension x dimension
		extra_center=False, 		Include extra component on central galaxy to correct for a color gradient
		color_sample_radius=1, 		When sampling colors from the original image, a (2*color_sample_radius + 1) x " box is used
		color_avg_p=1, 			For averaging in the color sampling, the l_(color_avg_p) norm is used
		gal_t=5e-4, 			Threshold the galaxy components
		jet_t=1e-2, 			Threshold the jet component
		fiber_radius=6, 		Radius (px) of spectroscopic fiber (for color gradient constraints)
		fiber_slope=1, 			Slope of logistic apodization for fiber mask (for color gradient constraints)
		galaxy_radius=50, 		Extent of galaxies involved
		extra_radius=20, 		Extent of extra peak component for color gradients
		jet_radius=40, 			Jet Extent
		general_slope=0.5, 		Logistic Apodization slope for the above
		color_others=True,		Fix other galaxies' colors at their color in the original image
		galaxy_constraint="M",		Deblending constraint for other galaxies
		bleed=None,			How much of the galaxy's spectrum the jet took on in the spectral splitting
		monotonicUseNearest=False, 	---Deblender parameters---
		max_iter=1000, 
		e_rel=[1e-6,1e-3], 
		traceback=False, 
		update_order=[1,0]
		"""
		objName = str(objParams[object_index][0])[:-1]
		filterWeights = np.zeros((3, len(bands)))
		filterWeights[0,4] = 1
		filterWeights[0,3] = 0.666
		filterWeights[1,3] = 0.333
		filterWeights[1,2] = 1
		filterWeights[1,1] = 0.333
		filterWeights[2,1] = 0.666
		filterWeights[2,0] = 1
		displayResults(data,result,
			writeFile=full, 
			folderName="Test13", 
			objName=objName, filterWeights=filterWeights, extra_center=extra_center, o=object_index, ks=[-1], use_psfs=False)
	except Exception, e:
		print("FAILED: " + str(e))
		if not full:
			traceback.print_exc()
