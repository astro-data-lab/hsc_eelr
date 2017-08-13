import numpy as np
import csv

read_data = open("objParams3.csv","r")
reader = csv.reader(read_data)
objNames = []
count = 0
for row in reader:
	if count > 0:
		objNames.append(row[0])
	count += 1

objData = open('crushVals.csv','w+')
for i in range(len(objNames)):
	print(objNames[i], i)
	s_data = open("%s/spec_decomposed.ecsv" % objNames[i],"r")
	full_spectrum = []
	count = 0
	for row in s_data:
		if count > 7:
			full_spectrum.append(np.fromstring(row, dtype=float, sep=' '))
		count += 1
	a_data = open("%s/sdss_xid.csv" % objNames[i],"r")
	reader = csv.reader(a_data)
	count = 0
	for row in reader:
		if count == 1:
			z_val = float(row[7])
		count += 1
	row_num = 0
	with open('%s/spec_mag.csv' % objNames[i], 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			row_num += 1
			if row_num == 6:
				SED_data = row
	h_alpha = (z_val+1)*6562.8

	spectrum = []	
	for i in range(len(full_spectrum)):
		if full_spectrum[i][0] > 9400 and full_spectrum[i][0] < 10800:
			spectrum.append(full_spectrum[i])
	spectrum = np.array(spectrum)

	if len(spectrum) > 5:
		within = spectrum[:,1] < 100000
		for t in range(25):
			std = np.std(spectrum[:,1][within])
			mean = np.mean(spectrum[:,1][within])
			within = np.bitwise_and(within, (spectrum[:,1] < mean + 3*std))
			within = np.bitwise_and(within, (spectrum[:,1] > mean - 3*std))

		int_width = 5
		peaks = []
		for i in range(int_width, len(spectrum)):
			power = 0
			for d in range(int_width):
				power += spectrum[i - d][1]
			if power > int_width*std*8:
				peaks.append(spectrum[i - int(d/2)][0])
		line_width = 100
		hii_power = 0
		tot_power = 0
		print(peaks)
		for i in range(len(spectrum)):
			if len(peaks) > 0:
				dist = np.amin(np.absolute(peaks - spectrum[i][0]))
			else:
				dist = 10000
			#print(spectrum[i][0], dist, spectrum[i][3]*1e19)
			if dist < line_width/2:
				hii_power += spectrum[i][3]				
			tot_power += spectrum[i][3]
		crush = np.minimum(1, 2*np.maximum(0, 1 - hii_power/tot_power))
	else:
		crush = 1
	#print(SED_data[30:35])
	print(crush)
	#print(lines)
	#print(high_sum)
	#print(low_sum)
	objData.write("{0} \n".format(crush))
	"""
	print((z_val+1)*6562.8)
	print(spectrum[-1][0])
	print("CONTINUUM")
	print(SED_data[25:30])
	print("JET")
	print(SED_data[30:35])
	print("CONTEXT")
	print(SED_data[35:40])
	"""
