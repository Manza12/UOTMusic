import numpy as np
import scipy.io.wavfile as wav
from parameters import *
import matplotlib.pyplot as plt
import util
from utilities import to_mono


def piptrack(S, sr=44100, threshold=0.1, sub_threshold=0.001):
    '''Pitch tracking on thresholded parabolically-interpolated STFT.

    This implementation uses the parabolic interpolation method described by [1]_.

    .. [1] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    S: np.ndarray [shape=(d,)]
        magnitude or power spectrum

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    threshold : float in `(0, 1)`
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`

    Returns
    -------
    pitches : np.ndarray [shape=(d,)]
    magnitudes : np.ndarray [shape=(d,)]
        Where `d` is the subset of FFT bins within `fmin` and `fmax`.

        `pitches[f]` contains instantaneous frequency at bin
        `f`

        `magnitudes[f]` contains the corresponding magnitudes.

        Both `pitches` and `magnitudes` take value 0 at bins
        of non-maximal magnitude.

    '''

    # Make sure we're dealing with magnitudes
    S = np.abs(S)


    n = len(S)
    n_fft = 2*(n-1)

    fft_freqs = np.linspace(0, float(sr) / 2, n, endpoint=True)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = np.zeros(n)
    shift = np.zeros(n)

    avg[1:-1] = 0.5 * (S[2:] - S[:-2])

    shift[1:-1] = 2 * S[1:-1] - S[2:] - S[:-2]

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

    # Compute the column-wise local max of S after thresholding
    max_S = np.max(S)
    # Find the argmax coordinates
    idx = np.argwhere(util.localmax(S * (S > (threshold_vect * max_S))))

    # Store pitch and magnitude
    pitches[idx] = ((idx + shift[idx]) * float(sr) / n_fft)

    mags[idx] = S[idx] + dskew[idx]

    pitches[-1] = 0.
    mags[-1] = 0.

    return pitches, mags


if __name__ == '__main__':
    file_name = 'do2'
    file_path = p.join(audio_path, file_name + '.wav')
    [fs_y, y] = wav.read(file_path)
    start_y = 0.01  # in seconds
    duration = 1  # in seconds
    end_y = start_y + duration  # in seconds
    y_segment = to_mono(y[int(fs_y * start_y): int(fs_y * end_y)])

    spectrum = np.abs(np.fft.fft(y_segment))
    spectrum_pos = spectrum[0:int(len(spectrum)/2)]
    pitches, mags = piptrack(spectrum, sr=44100, threshold=1e4, sub_threshold=0.001)

    plt.figure(1)
    plt.plot(spectrum_pos)
    plt.xscale('log')
    plt.scatter(pitches, mags)

    plt.show()
