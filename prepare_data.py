import numpy as np
import os, fitsio
from sys import argv
import deblender
from plotting import plotColorImage
from deblender.psf_match import matchPSFs

# get difference kernels for all PSFs
if __name__ == "__main__":

    if len(argv) == 2:
        objname = argv[1]
        dirs = [objname]
    else:
        from glob import glob
        dirs = glob('SDSS*')

    bands = ['g','r','i','z','y']
    B = len(bands)
    pixel_scale = 1.
    radius_cut = 10.
    fwhm = 1.7

    contrast = 1e5
    filterWeights = np.zeros((3, len(bands)))
    filterWeights[0,4] = 1
    filterWeights[0,3] = 0.666
    filterWeights[1,3] = 0.333
    filterWeights[1,2] = 1
    filterWeights[1,1] = 0.333
    filterWeights[2,1] = 0.666
    filterWeights[2,0] = 1

    for objname in dirs:
        print objname
        psf_bands = []
        for b in range(B):
            psf = fitsio.FITS("%s/psf-%s.fits" % (objname, bands[b]))
            psf_bands.append(psf[0].read())
            psf.close()

        kernels, diff_kernels, reconv_kernels, psf_error = deblender.psf_match.matchPSFs(psf_bands, fwhm=fwhm, pixel_scale=pixel_scale, radius_cut=radius_cut)

        plotColorImage(kernels, filterWeights=filterWeights, contrast=contrast, title=objname)
        plotColorImage(reconv_kernels, filterWeights=filterWeights, contrast=contrast, title=objname+" model")
        plotColorImage(kernels - reconv_kernels, filterWeights=filterWeights, contrast=contrast, title=objname)

        # save difference kernels
        for i in range(len(bands)):
            filename = "%s/psf-diff_kernel_%s.fits" % (objname, bands[i])
            fitsio.write(filename, diff_kernels[i], clobber=True)
