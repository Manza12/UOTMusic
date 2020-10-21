import numpy as np
import scipy.io.wavfile as wav
from parameters import *
import matplotlib.pyplot as plt
import util
from utilities import to_mono


def piptrack(spectrum, sr=44100, threshold=0.1, sub_threshold=0.001):
    """ Pitch tracking on thresholded parabolically-interpolated STFT.

        This implementation uses the parabolic interpolation method described by [1].
        .. [1] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

        Parameters
        ----------
        spectrum: numpy array [shape=(d,)]
            magnitude or power _spectrum
        sr: number > 0 [scalar]
            audio sampling rate of `y`
        threshold : float in `(0, 1)`
            A bin in _spectrum X is considered a pitch when it is greater than
            `threshold*X.max()`
        sub_threshold: ToDo

        Returns
        ----------
        pitches, mags: numpy arrays [shape=(d,)]
            Where `d` is the subset of FFT bins within `fmin` and `fmax`.
            `pitches[f]` contains instantaneous frequency at bin
            `f`
            `mags[f]` contains the corresponding magnitudes.
            Both `pitches` and `mags` take value 0 at bins
            of non-maximal magnitude.
    """

    # Make sure we're dealing with magnitudes
    spectrum = np.abs(spectrum)

    n = len(spectrum)
    n_fft = 2*(n-1)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = np.zeros(n)
    shift = np.zeros(n)

    avg[1:-1] = 0.5 * (spectrum[2:] - spectrum[:-2])

    shift[1:-1] = 2 * spectrum[1:-1] - spectrum[2:] - spectrum[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift[1:-1] = avg[1:-1] / (shift[1:-1] + (np.abs(shift[1:-1]) < util.tiny(shift[1:-1])))

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros(n)
    mags = np.zeros(n)

    # Compute the threshold vector
    threshold_vect = threshold / 10**(5*np.arange(n) / (n-1))

    threshold_vect[int(n * np.log10(threshold / sub_threshold) / 5):] = sub_threshold

    # Compute the column-wise local max of _spectrum after thresholding
    max_spectrum = np.max(spectrum)
    # Find the argmax coordinates
    idx = np.argwhere(util.localmax(spectrum * (spectrum > (threshold_vect * max_spectrum))))

    # Store pitch and magnitude
    pitches[idx] = ((idx + shift[idx]) * float(sr) / n_fft)

    mags[idx] = spectrum[idx] + dskew[idx]

    pitches[-1] = 0.
    mags[-1] = 0.

    return pitches, mags


if __name__ == '__main__':
    file_name = 'do2'
    file_path = p.join(audio_path, file_name + '.wav')
    [fs_y, y] = wav.read(file_path)
    y_float = y / np.iinfo(np.int16).max
    start_y = 0.01  # in seconds
    duration = 1  # in seconds
    end_y = start_y + duration  # in seconds
    y_segment = to_mono(y_float[int(fs_y * start_y): int(fs_y * end_y)])

    _spectrum = np.abs(np.fft.fft(y_segment)) * (2 / len(y_segment))
    spectrum_pos = _spectrum[0:len(_spectrum) // 2 + 1]
    _pitches, _mags = piptrack(spectrum_pos, sr=44100, threshold=.1, sub_threshold=0.001)

    freqs = np.linspace(0, fs_y / 2, int(len(_spectrum) // 2) + 1, endpoint=True)

    plt.figure(1)
    plt.plot(freqs, spectrum_pos)
    plt.xscale('log')
    plt.scatter(_pitches[np.nonzero(_pitches)], _mags[np.nonzero(_pitches)], color='green')

    plt.show()
