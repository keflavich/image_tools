"""
Trivial clean implementation
"""
import numpy as np
#from astropy.nddata import convolve

def clean(s, g, niter=100, gain=0.1, threshold=1.25):
    """
    A trivial 1D implementation of the CLEAN algorithm
    """
    g = g/g.max()
    result = np.zeros_like(s)
    kern = np.zeros_like(s)
    mid = kern.size/2
    kern[mid] = 1
    sig = np.convolve(kern,g,mode='same')
    for ii in xrange(niter):
        peak = s.argmax()
        mx = s[peak]
        result[peak] += mx*gain
        sig *= mx*gain
        sig = np.roll(sig,peak-mid)
        s = s - sig
        sig /= mx*gain
    return result,s
