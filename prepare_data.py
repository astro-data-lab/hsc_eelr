import numpy as np
import galsim
import os

def getDiffKernel(P, P0, nx=50, ny=50):
    PD = galsim.Convolve(P, galsim.Deconvolve(P0))
    PDimg = PD.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
    return PDimg

def getDiffKernels(psf_bands, P0, thresh=5e-3):
    kernels = []
    diff_kernels = []
    reconv_kernels = []
    psf_error = 0

    for i in range(len(psf_bands)):
        P = psf_bands[i]
        nx,ny = P.image.array.shape

        PDimg = getDiffKernel(P, P0, nx=nx, ny=ny)
        kernels.append(P.image.array)
        diff_kernels.append(PDimg.array)

        # just for testing
        P0img = P0.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
        P0FromImage = galsim.InterpolatedImage(P0img)

        # truncate difference kernel below 3e-3
        # to make the more compact
        mask = np.abs(PDimg.array) < thresh
        PDimg.array[mask] = 0

        PDFromImage = galsim.InterpolatedImage(PDimg)
        P0PD = galsim.Convolve(P0FromImage, PDFromImage)
        P0PDimg = P0PD.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
        reconv_kernels.append(P0PDimg.array)
        error = ((reconv_kernels[i]-kernels[i])**2).sum()
        psf_error = max(psf_error, error)
        #print fwhm, bands[i], error, nx*ny - mask.sum()
    return kernels, diff_kernels, reconv_kernels, psf_error

# get difference kernels for all PSFs
from sys import argv
objname = argv[1]

import fitsio
bands = ['g','r','i','z','y']
psf_bands = []
for b in bands:
    psf = galsim.fits.read("%s/psf-%s.fits" % (objname, b))
    psf_bands.append(galsim.InterpolatedImage(psf))

# reference kernel
pixel_scale = 1.
psf_thresh = 5e-3
fwhms = np.arange(1., 3.2, 0.2)
psf_errors = []
for fwhm in fwhms:
    P0 = galsim.Gaussian(fwhm=fwhm)
    kernels, diff_kernels, reconv_kernels, psf_error = getDiffKernels(psf_bands, P0, thresh=psf_thresh)
    psf_errors.append(psf_error)

# get kernels with FWHM that has smallest pixelation errors
min_psf = np.argmin(psf_errors)
print "best overall PSF error: %r at FWHM = %.1f" % (psf_errors[min_psf], fwhms[min_psf])
fwhm = fwhms[min_psf]
P0 = galsim.Gaussian(fwhm=fwhm)
#P0img = P0.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
kernels, diff_kernels, reconv_kernels, psf_error = getDiffKernels(psf_bands, P0)

# save difference kernels
for i in range(len(bands)):
    filename = "%s/psf-diff_kernel_%s.fits" % (objname, bands[i])
    fitsio.write(filename, diff_kernels[i])

"""
# make a bands-combined coadds for detection
os.system("mkdir multiband-coadds")
imgs = []
for b in bands:
    hdu = fitsio.FITS("%s/stamp-%s.fits" % (objname, b))
    imgs.append(hdu[0][:,:])
sum_imgs = np.sum(imgs, axis=0)
filename = "multiband-coadds/%s.fits" % objname
fitsio.write(filename, sum_imgs)
"""
