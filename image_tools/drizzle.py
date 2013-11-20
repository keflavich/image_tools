import numpy

def masktozero(arr):
    """
    Convert all NANs or masked elements of an array to zero, and return a numpy
    array (NOT a masked array).
    """
    if hasattr(arr,'mask'):
        arr[arr.mask] = 0
    else:
        arr[arr!=arr] = 0

    return numpy.array(arr)

def drizzle(tstomap,ts,mapshape,weights=1,weightmap=None):
    """
    Drizzle a timestream onto a map.  Returns the map of the weighted average
    per pixel of the input timestream.
    (note that this works for any 1D array with a same-size mapping to an
    image; I've written it with timestreams in mind though)

    tstomap - mapping from timestream -> map.  len(tstomap) = len(ts)
        Both tstomap and ts should be one-dimensional, but they'll be raveled
        if you don't do it yourself
    weights - needs to have the same dimensions as ts *or* be scalar
            (default=1)
    mapshape - [nx,ny] simple 2D map specification.  Make sure your map
        includes all points mapped to

    You can specify a weightmap to increase efficiency instead of computing it
    """
    newmap = numpy.zeros(mapshape)
    # don't need to mask out when adding zero; simplifies indexing to have no masks
    ts_to_index = masktozero((ts*weights).ravel())

    if len(tstomap.shape) > 1: # tstomap must be flat
        tstomap = tstomap.ravel()

    tsmapped = numpy.bincount(tstomap,ts_to_index)

    # bincount has length = argmax(tstomap), but the x/y index arrays
    # go all the way to the full map size.  Append zeros to fill
    maxind = mapshape[0]*mapshape[1]
    if tsmapped.shape[0] < maxind:
        tsmapped = numpy.concatenate([tsmapped,
            numpy.zeros(maxind-tsmapped.shape[0])])
    xinds,yinds = (a.ravel() for a in numpy.indices(mapshape))
    newmap[xinds,yinds] = tsmapped

    # do the same for weights unless a weightmap is specified
    if weightmap is None: 
        wm = numpy.zeros(mapshape)
        if numpy.isscalar(weights): 
            weights_to_index = numpy.ones(ts_to_index.shape)*weights
        else:
            weights_to_index = masktozero((weights).ravel())

        tsweights = numpy.bincount(tstomap,weights_to_index)
        if tsweights.shape[0] < maxind:
            tsweights = numpy.concatenate([tsweights,
                numpy.zeros(maxind-tsweights.shape[0])])
        wm[xinds,yinds] = tsweights
    else:
        wm = weightmap

    return newmap/wm
