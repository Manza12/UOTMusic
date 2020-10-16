import numpy as np
from numba import jit

def tiny(x):
    '''Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in `x`'s
    data type (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is `x.dtype`.

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of `x`.
        If `x` is integer-typed, then the tiny value for `np.float32`
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------

    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    '''

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny




@jit(nopython=True)
def localmax(x):
    """Find local maxima in an array `x`.

    An element `x[i]` is considered a local maximum if the following
    conditions are met:

    - `x[i] > x[i-1]`
    - `x[i] >= x[i+1]`

    Note that the first condition is strict, and that the first element
    `x[0]` will never be considered as a local maximum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    Parameters
    ----------
    x     : np.ndarray [shape=(d,)]
      input vector

    Returns
    -------
    m     : np.ndarray [shape=(d,), dtype=bool]

    """

    #Create boolean array filled with False
    localmax_array = np.zeros_like(x,dtype=np.bool_)

    #Check for each index exept first and last if it is a local maximum
    localmax_array[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])


    return localmax_array







