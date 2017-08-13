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

def loadData(objParams, dimension=119, galaxy_constraint = "M", p_buffer=15):
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
	projected = (peaks[:,0] > -p_buffer) & (peaks[:,1] > -p_buffer) & (peaks[:,0] < data.shape[1] + p_buffer) & (peaks[:,1] < data.shape[1] + p_buffer)
	peaks = peaks[projected]
	for i in range(peaks.shape[0]):
		peaks[i][0] = np.minimum(np.maximum(peaks[i][0], 0), data.shape[1] - 1)
		peaks[i][1] = np.minimum(np.maximum(peaks[i][1], 0), data.shape[1] - 1)

	constraints = [constraints[i] for i in range(len(constraints)) if projected[i] == 1]
	
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
	#load given colors
	gal_sed = np.array([float(SED_data[35]),float(SED_data[36]),float(SED_data[37]),float(SED_data[38]),float(SED_data[39])])
	jet_sed = np.array([float(SED_data[30]),float(SED_data[31]),float(SED_data[32]),float(SED_data[33]),float(SED_data[34])])
	
	#gal_sed = np.array([float(SED_data[35]),float(SED_data[36]),float(SED_data[37]),float(SED_data[38]),float(SED_data[39])])
	
	# manually overriding found y-band value (seems to often be corrupted by galaxy lines)
	#jet_sed = np.array([0.08,0.17,0.55,0,0])#colorSample(data,32,9, color_sample_radius=0)#np.array([0.08094098, 0.11424346, 0.48816309, 0.1028126, 0.21383987])
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

def defineParameters(data, peaks, jet_sed, gal_sed, color_sample_radius=1, color_avg_p=1, gal_t=5e-4, jet_t=1e-2, color_others=True, close_color=False, raise_gal_y=False, crush_jet_y=1, objName=""):
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
	
	# Color corrections:
	bleed = np.array([0.,0.,0.,0.,0.])
	# Given galaxy y-band can be too small if spectral data stops early
	if raise_gal_y:	
		gal_sed[4] = np.maximum(gal_sed[3], gal_sed[4])
	# don't expect jet to have much y-band flux at all (Galaxy H-alpha is messing with it at these z-values)
	bleed[4] = jet_sed[4]
	gal_sed2 = gal_sed + bleed
	jet_sed2 = jet_sed - bleed
	pal = np.zeros_like(data)
	gradient = np.linspace(0,2,data.shape[1])
	pal += gradient[None,:,None]*gal_sed[:,None,None]
	pal += (1 - gradient[None,:,None])*gal_sed2[:,None,None]
	#plotColorImage(pal, objName=(objName + "_0GColors"), folderName="Test25", writeFile=True)

	pal = np.zeros_like(data)
	pal += gradient[None,:,None]*jet_sed[:,None,None]
	pal += (1 - gradient[None,:,None])*jet_sed2[:,None,None]
	#plotColorImage(pal, objName=(objName + "_0JColors"), folderName="Test25", writeFile=True)

	bleed[4] = (1 - crush_jet_y)*jet_sed[4]
	gal_sed += bleed
	jet_sed -= bleed
	SEDs[0] = gal_sed
	SEDs[-1] = jet_sed
	for i in range(SEDs.shape[0]):
		SEDs[i] = proxmin.operators.prox_unity_plus(SEDs[i], 1)
	
	if close_color:
		min_dist = 10000
		min_index = 0
		for i in range(1,SEDs.shape[0]-1):
			#curr_dist = ((SEDs[0] - SEDs[i])**2).sum()
			curr_dist = ((SEDs[0,0:4] - SEDs[i,0:4])**2).sum() # y-band is messed up
			if curr_dist < min_dist:
				min_dist = curr_dist
				min_index = i
		pull = 1 # < 1 to trust the spectral data somewhat
		SEDs[0] += (SEDs[min_index] - SEDs[0])*pull

	# create thresholds
	l1_thresh = np.ones(len(peaks))*gal_t
	l1_thresh[-1] = jet_t
	return weights, SEDs, l1_thresh

def defineConstraints(data, peaks, SEDs, extra_center=False, l1_thresh=None, fiber_radius=6, fiber_slope=1, galaxy_radius=50, central_radius=None, extra_radius=20, jet_radius=40, general_slope=0.5, color_give=None):
	
	# define constraints
	fiber_mask = np.zeros(data.shape[1]**2)
	temp = ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[:,None] + ((np.arange(data.shape[1]) - data.shape[1]/2)**2)[None,:]
	fiber_mask = (1/(1 + np.exp(fiber_slope*(temp**0.5 - fiber_radius)))).T.ravel()
	fiber_mask = fiber_mask/fiber_mask.sum()
		
	if color_give is None:
		color_give = np.zeros(5)

	def prox_SED(A, step, SEDs=None, Xs=None, extra_center=False, use_absolute=False, color_give=None):
		if extra_center and (step < 0.0005):
			S = Xs[1]
			b = A.shape[0]
			k = A.shape[1]
			# record power of each component going through fiber
			spec_mag = np.zeros((k,b)) # flux from each component through fiber
			for i in range(k):
				model = np.dot(A[:, i:(i+1)], S[i:(i+1), :])
				model *= fiber_mask[None, :]
				fiber_sum = model.sum(axis=1)
				spec_mag[i] = fiber_sum
			model = np.dot(A[:,0:2], S[0:2,:])
			model *= fiber_mask[None,:]
			#print(model.shape)
			fiber_sum = proxmin.operators.prox_unity_plus(model.sum(axis=1), 1)
	
			# color flexibility in bands specified by color_give
			push_back = 1/(1 + color_give)
			galaxy_below = A[:,0].T < SEDs[0]
			push_back[galaxy_below] = 1 # if the galaxy has below expected flux, force
			A[:,0:2] -= ((fiber_sum - SEDs[0])*push_back)[:,None]
			A[:,0] = proxmin.operators.prox_unity_plus(A[:,0], 1)
			A[:,1] = proxmin.operators.prox_unity_plus(A[:,1], 1)
			
		    	for i in range(2, len(A[0])):
				if not math.isnan(SEDs[i][0]):
					A[:,i] = SEDs[i]	
			if use_absolute:
				# Todo (really should be in prox_S)
				model = np.dot(A[:,0:2], S[0:2,:])
				model *= fiber_mask[None,:]
				fiber_sum = model.sum(axis=1)
		
				model = np.dot(A[:,-1:], S[-1:,:])
				model *= fiber_mask[None,:]
				fiber_sum = model.sum(axis=1)
		else:
			for i in range(1,len(A[0])):
				if not math.isnan(SEDs[i][0]):
					A[:,i] = SEDs[i]

			# color flexibility in bands specified by color_give
			push_back = 1/(1 + color_give)
			galaxy_below = A[:,0].T < SEDs[0]
			push_back[galaxy_below] = 1 # if the galaxy has below expected flux, force
			A[:,0] += (SEDs[0,:].T - A[:,0])*push_back

		return proxmin.operators.prox_unity_plus(A, step, axis=0)
	prox_A = partial(prox_SED, SEDs=SEDs, extra_center=extra_center, color_give=color_give)

	# define masks for localizing jet/galaxies
	radii = np.ones(len(peaks))*galaxy_radius
	if extra_center:
		radii[1] = extra_radius
	radii[-1] = jet_radius
	if central_radius is None:
		central_radius = galaxy_radius
	radii[0] = central_radius
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
	return color

def deblend(objParams, readParams=False, dimension=119, extra_center=False, color_sample_radius=1, color_avg_p=1, gal_t=5e-4, jet_t=1e-2, fiber_radius=6, fiber_slope=1, galaxy_radius=50, central_radius=None, extra_radius=20, jet_radius=40, general_slope=0.5, color_others=True, color_give=None, close_color = False, raise_gal_y=False, crush_jet_y=1, galaxy_constraint="M", p_buffer=15, monotonicUseNearest=False, max_iter=1000, e_rel=[1e-6,1e-3], traceback=False, update_order=[1,0], objName=""):
	
	if readParams:
		dimension=objParams[2]			#Final Image will be dimension x dimension
		extra_center=objParams[3]		#Include extra component on central galaxy to correct for a color gradient
		color_sample_radius=objParams[4]	#When sampling colors from the original image, a (2*color_sample_radius + 1) x " box is used
		color_avg_p=objParams[5]		#For averaging in the color sampling, the l_(color_avg_p) norm is used
		gal_t=objParams[6]			#Threshold the galaxy components
		jet_t=objParams[7]			#Threshold the jet component
		p_buffer=objParams[8]	
		fiber_radius=objParams[9]		#Radius (px) of spectroscopic fiber (for color gradient constraints)
		fiber_slope=objParams[10]		#Slope of logistic apodization for fiber mask (for color gradient constraints)
		galaxy_radius=objParams[11] 		#Extent of galaxies involved
		central_radius=objParams[12]		#Extent of central galaxy, None: same as others
		extra_radius=objParams[13]		#Extent of extra peak component for color gradients
		jet_radius=objParams[14] 		#Jet Extent
		general_slope=objParams[15]		#Logistic Apodization slope for the above
		color_others=objParams[16]		#Fix other galaxies' colors at their color in the original image
		color_give=objParams[17]		#Allow for flexibility in fitting A to SEDs
		close_color=objParams[18]		#Set Central Galaxy Color to the other galaxy in the image closest in color-space
		raise_gal_y=objParams[19]		#Force Galaxy's y-band to be greater than or equal to r-band (cases where spectral data stops before y-bnad)
		crush_jet_y=objParams[20]		#Force Jet's y-band to be 0 (when H-alpha is picked up as jet color)
		galaxy_constraint=objParams[21]		#Deblending constraint for galaxies
		monotonicUseNearest=objParams[22] 	#---Deblender parameters---
		max_iter=objParams[23]
		e_rel=objParams[24]
		traceback=objParams[25]
		update_order=objParams[26]	

	data, psfs, peaks, constraints, jet_sed, gal_sed = loadData(objParams, dimension=dimension, galaxy_constraint=galaxy_constraint, p_buffer=p_buffer)

	peaks, constraints = processData(peaks, constraints, data, extra_center=extra_center)

	weights, SEDs, l1_thresh = defineParameters(data, peaks, jet_sed, gal_sed, color_sample_radius=color_sample_radius, color_avg_p=color_avg_p, gal_t=gal_t, jet_t=jet_t, color_others=color_others, close_color=close_color, raise_gal_y=raise_gal_y, crush_jet_y=crush_jet_y, objName=objName)
	
	prox_A, prox_S = defineConstraints(data, peaks, SEDs, extra_center=extra_center, l1_thresh=l1_thresh, fiber_radius=fiber_radius, fiber_slope=fiber_slope, galaxy_radius=galaxy_radius, central_radius=central_radius, extra_radius=extra_radius, jet_radius=jet_radius, general_slope=general_slope, color_give=color_give)
	
	#print(SEDs)	

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
	
	#print(A.T)
	
	return result, data

# load stored object-deblending parameters
read_data = open("objParams3.csv","r")
reader = csv.reader(read_data)
objParams = []
count = 0
for row in reader:
	temp_peaks = np.fromstring(row[1], dtype=int, sep=' ')
	types = "ibifffifffffffbzbbfsbizby"
	if count > 0:
		row_data = [row[0],np.flip(temp_peaks.reshape((len(temp_peaks)/2, 2)),axis=1)]
		for i in range(len(types)):
			curr = types[i:i+1]
			if curr == "i":
				row_data.append(int(row[i+4]))
			if curr == "b":
				row_data.append(bool(int(row[i+4])))
			if curr == "f":
				row_data.append(float(row[i+4]))
			if curr == "s":
				row_data.append(str(row[i+4]))
			if curr == "y":
				row_data.append(np.fromstring(row[i+4], dtype=int, sep=' '))
			if curr == "z":
				row_data.append(np.fromstring(row[i+4], dtype=float, sep=' '))
		objParams.append(row_data)
	count += 1

# set the object number for testing------
full = False
if full:
	obj_nums = np.arange(len(objParams))
else:
	obj_nums = [14]

# process objects
for object_index in obj_nums:
	try:
		objName = str(objParams[object_index][0])[:-1]
		print("OBJECT: {0}, #{1}, {2}x{2} for {3} iterations".format(objName, object_index, objParams[object_index][2], objParams[object_index][23]))
		extra_center=True
		result, data = deblend(objParams[object_index], 
			readParams=True,
			extra_center=extra_center, 
			max_iter=1000,
			galaxy_radius=15,
			central_radius=40,
			extra_radius=15,
			general_slope=1,
			jet_radius=20,
			jet_t=1e-3,
			galaxy_constraint="M",
			p_buffer=15,
			color_give=np.array([0.,0.,0.,0.,0.]),
			close_color=True,
			objName=objName,
			dimension=85)
		"""
		objParams,			objParams[0]: Object Name objParams[1]: Nx2 numpy array of image peaks
		readParams,			Tell deblender to take object parameters from sheet instead of those passed
		dimension=119, 			Final Image will be dimension x dimension
		extra_center=False, 		Include extra component on central galaxy to correct for a color gradient
		color_sample_radius=1, 		When sampling colors from the original image, a (2*color_sample_radius + 1) x " box is used
		color_avg_p=1, 			For averaging in the color sampling, the l_(color_avg_p) norm is used
		gal_t=5e-4, 			Threshold the galaxy components
		jet_t=1e-2, 			Threshold the jet component
		p_buffer=15,			Range where peaks are projected onto trimmed image
		fiber_radius=6, 		Radius (px) of spectroscopic fiber (for color gradient constraints)
		fiber_slope=1, 			Slope of logistic apodization for fiber mask (for color gradient constraints)
		galaxy_radius=50, 		Extent of galaxies involved
		central_radius=None, 		Extent of central galaxy, None: same as others
		extra_radius=20, 		Extent of extra peak component for color gradients
		jet_radius=40, 			Jet Extent
		general_slope=0.5, 		Logistic Apodization slope for the above
		color_others=True,		Fix other galaxies' colors at their color in the original image
		color_give=None,		Allow for flexibility in fitting A to SEDs
		close_color=False,		Set Central Galaxy Color to the other galaxy in the image closest in color-space
		raise_gal_y=False,		#Force Galaxy's y-band to be greater than or equal to r-band (cases where spectral data stops before y-bnad)
		crush_jet_y=1,			#Multiplier of Jet's y-band (when H-alpha is picked up as jet color)
		galaxy_constraint="M",		Deblending constraint for other galaxies
		monotonicUseNearest=False, 	---Deblender parameters---
		max_iter=1000, 
		e_rel=[1e-6,1e-3], 
		traceback=False, 
		update_order=[1,0]
		"""
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
			folderName="Test25", 
			objName=objName, filterWeights=filterWeights, extra_center=extra_center, o=object_index, ks=[-1], use_psfs=False)
	except Exception, e:
		print("FAILED: " + str(e))
		if not full:
			traceback.print_exc()
