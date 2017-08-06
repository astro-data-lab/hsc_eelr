from __future__ import division
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import csv
from scipy.ndimage.filters import gaussian_filter

numObjects = -1
obj_names = []
with open('objNames.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		numObjects += 1
		if numObjects > 0:
			obj_names.append(row[0])

# read data to be potentially changed
objParams = [None] * numObjects
count = 0
with open('objParams2.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		if count < numObjects:
			objParams[count] = row
			count += 1
objData = open('objParams2.csv','w+')

# so that the data isn't lost
try: 
	mode = 1
	if mode == 0:
		start_num = 0
		nrows = 6
		ncols = 7
		fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,8))
		for r in range(nrows):
			for c in range(ncols):		
				index = start_num + c + r*ncols
				f = misc.imread(obj_names[index] + 'color_stamp-riz.png')
				axs[r,c].set_title(index)
				axs[r,c].imshow(f)
		fig.subplots_adjust(hspace=0.5, wspace=0.3)
	else:
		# hand-tuned
		mults = [10,12,10,11, 11,14,15,12, 16,16,16,20,
		10,10,12,10, 10,12,12,10, 14,10,12,15,
		14,10,16,20, 8,10,16,16, 16,16,10,10,
		12,16,16,16, 12,20,8,14, 14,12,12,10,
		20,12,12,10, 14,8,10,14, 12,16,16,16,
		12,14,12,14, 14,10,10,9, 8,12,12,14,
		12,12,14,12, 10,10,10,20, 16,12,14,10]

		start_num = 0
		nrows = 7
		ncols = 12

		fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,12))
		for r in range(nrows):
			for c in range(ncols):	
				obj = start_num + c + r*ncols
				if obj != 12:
					path = obj_names[obj]
					print(obj)
					bands = []
					bands.append(fits.open(path + 'stamp-g.fits')[0].data)
					bands.append(fits.open(path + 'stamp-r.fits')[0].data)
					bands.append(fits.open(path + 'stamp-i.fits')[0].data)
					bands.append(fits.open(path + 'stamp-z.fits')[0].data)
					bands.append(fits.open(path + 'stamp-y.fits')[0].data)
					pic = misc.imread(path + 'color_stamp-riz.png')		
				
					weighting = np.array([0,0,0,5,0])[:,None,None]			
					total = np.sum(bands*weighting,axis=0)
					peaks = []

					# Find threshold level
					average = 0
					p = 1/2
					mult = 10
					for i in range(len(total)):
						for j in range(len(total[0])):
							if total[i][j] > 0:
								average += total[i][j]**p
					average /= total.shape[0]*total.shape[1]
					average **= 1/p
					average *= mults[obj]

					extent = 3
					check_range = 5
					region_mult = 1
					for i in range(len(total)):
						for j in range(len(total[0])):
							max = -100
							for ii in range(-extent,extent+1):
								for jj in range(-extent,extent+1):
									try:
										max =  np.maximum(total[i+ii][j+jj],max)
									except:
										pass
							if total[i][j] == max:
								surround = 0
								num_surround = 0
								region = 100
								num_region = 0
								for ii in range(-check_range,check_range+1):
									for jj in range(-check_range,check_range+1):
										try:
											region = np.minimum(region, total[i+ii][j+jj])
											num_region += 1	
										except:
											pass
								if (max > (region_mult*average + region/num_region)):
									peaks.append(np.array([i,j]))

					peaks = np.array(peaks)
					if nrows > 1:
						axs[r][c].imshow(pic)
						axs[r][c].scatter(peaks[:,1],119 - peaks[:,0], alpha=0.5, s=8, color="blue")
						axs[r][c].set_title(obj)
					else:
						axs.imshow(pic)
						axs.scatter(peaks[:,1],119 - peaks[:,0], alpha=0.5, s=20, color="blue")
						axs.set_title(obj)
					peak_string = ""
					for i in range(len(peaks)):
						peak_string += " " + str(peaks[i][0]) + " " + str(peaks[i][1])
					weights_string = ""
					for i in range(len(weighting)):
						weights_string += " " + str(weighting[i][0][0])
					objParams[obj] = []
					objParams[obj].append(obj_names[obj])
					objParams[obj].append(peak_string)
					objParams[obj].append(mults[obj])
					objParams[obj].append(weights_string)
except Exception, e:
	print(e)
	pass
# rewrite data
for i in range(numObjects):
	try:
		objData.write("{0},{1},{2},{3}\n".format(objParams[i][0],objParams[i][1],objParams[i][2],objParams[i][3]))
	except:
		pass
		objData.write("\n")

fig.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

objData.close()

"""
f = misc.imread('HumVI_riz.png') # 
uses the Image module (PIL)
print(f.shape)
image1 = fits.open('stamp-g.fits')[0].data
image2 = fits.open('stamp-r.fits')[0].data
image3 = fits.open('stamp-i.fits')[0].data
image4 = fits.open('stamp-z.fits')[0].data
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
axs[0,0].set_title("PSF-g")
axs[0,0].imshow(image1)
axs[0,1].set_title("PSF-r")
axs[0,1].imshow(image2)
axs[1,0].set_title("PSF-i")
axs[1,0].imshow(image3)
axs[1,1].set_title("PSF-z")
axs[1,1].imshow(image4)
plt.show()
"""
