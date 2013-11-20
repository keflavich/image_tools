import fast_ffts
import warnings
import numpy as np
import shift

def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
    """
    *translated from matlab*
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

    Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    a small region.
    usfac         Upsampling factor (default usfac = 1)
    [nor,noc]     Number of pixels in the output upsampled DFT, in
                  units of upsampled pixels (default = size(in))
    roff, coff    Row and column offsets, allow to shift the output array to
                  a region of interest on the DFT (default = 0)
    Recieves DC in upper left corner, image center must be in (1,1) 
    Manuel Guizar - Dec 13, 2007
    Modified from dftus, by J.R. Fienup 7/31/06

    This code is intended to provide the same result as if the following
    operations were performed
      - Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      - Take the FFT of the larger array
      - Extract an [nor, noc] region of the result. Starting with the 
        [roff+1 coff+1] element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy.fft import ifftshift
    from numpy import pi,newaxis,floor

    nr,nc=np.shape(inp);
    # Set defaults
    if noc is None: noc=nc;
    if nor is None: nor=nr;
    # Compute kernels and obtain DFT by matrix products
    kernc=np.exp((-1j*2*pi/(nc*usfac))*( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )*( np.arange(noc) - coff  )[newaxis,:]);
    kernr=np.exp((-1j*2*pi/(nr*usfac))*( np.arange(nor).T - roff )[:,newaxis]*( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]);
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    out=np.dot(np.dot(kernr,inp),kernc);
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    return out 

def upsample_image(image, upsample_factor=1, output_size=None, nthreads=1, use_numpy_fft=False,
        xshift=0, yshift=0):
    """
    Use dftups to upsample an image (but takes an image and returns an image with all reals)
    """
    fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    imfft = ifftn(image)

    if output_size is None:
        s1 = image.shape[0]*upsample_factor
        s2 = image.shape[1]*upsample_factor
    elif hasattr(output_size,'__len__'):
        s1 = output_size[0]
        s2 = output_size[1]
    else:
        s1 = output_size
        s2 = output_size

    ups = dftups(imfft, s1, s2, upsample_factor, roff=yshift, coff=xshift)

    return np.abs(ups)

def dftups1d(inp,nor=None,usfac=1,roff=0):
    """
    1D upsampling... not exactly dft becuase I still don't understand it =(
    """
    # this function is translated from matlab, so I'm just going to pretend
    # it is matlab/pylab
    from numpy.fft import ifftshift
    #from numpy import pi,newaxis,floor
    from scipy.signal import resample


    nr=np.size(inp);
    newsize = nr * usfac
    #shifted = shift(inp, roff, mode='wrap')
    shifted = shift.shift1d(inp,roff)
    ups = resample(shifted.astype('float'),newsize)
    lolim = nr/2-nr/2
    uplim = nr/2+nr/2
    # I think it would always have to be wrong on the upper side
    if uplim-lolim > nr:
        uplim -= 1
    elif uplim-lolim < nr:
        uplim += 1
    if uplim - lolim != nr: raise ValueError('impossible?')
    out = ups[lolim:uplim]

    #oldx = np.arange(nr)
    #newx = np.linspace(nr/2.-nr/2./usfac+roff/usfac,nr/2.+nr/2./usfac+roff/usfac,nr)
    #oldx = np.linspace(0,1,nr)
    #newx = np.linspace(0,1,newsize)
    #inshift = shift.shift1d(inp,roff)
    #out = ups = np.interp(newx,oldx,np.real(inp))

    #lolim = newsize/2+roff*usfac-nr/2
    #uplim = newsize/2+roff*usfac+nr/2
    #out = ups[lolim:uplim]
    
    # Set defaults
    #if nor is None: nor=nr;
    # Compute kernels and obtain DFT by matrix products
    #kernc=np.exp((-1j*2*pi/(nc*usfac))*( ifftshift(np.arange(nc) - floor(nc/2)).T[:,newaxis] )*( np.arange(noc) - coff  )[newaxis,:]);
    #kernr=np.exp((-1j*2*pi/(nr*usfac))*( np.arange(nor).T - roff )[:,newaxis]*( ifftshift(np.arange(nr)) - floor(nr/2) )[newaxis,:]);
    #kernc=np.ones(nr,dtype='float')/float(nr)
    #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
    #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
    #out=np.dot(kernr,inp)
    #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
    return out 
