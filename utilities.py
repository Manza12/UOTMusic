from pathlib import Path
from parameters import *


def create_data(freqs, mags):
    return np.array(freqs), np.array(mags)


def note2freq(note, reference_frequency=REFERENCE_FREQUENCY):
    freq = reference_frequency * np.power(2, note / 12)
    return freq


def freq2note(freq, reference_frequency=REFERENCE_FREQUENCY):
    note = 12 * np.log2(freq / reference_frequency)
    return note


def to_mono(y):
    if np.size(y, 1) == 2:
        return (y[:, 0] + y[:, 1]) / 2
    elif np.size(y, 1) == 1:
        return y
    else:
        raise ValueError("To mono nor stereo.")


def create_dir(dir_name):
    p = Path(dir_name)
    p.mkdir(parents=True, exist_ok=True)


def create_random_dirac(n, m):
    xs, ys = np.sort(np.random.uniform(0, 1, n)), np.sort(np.random.uniform(0, 1, m))  # random positions
    p, q = np.random.uniform(0, 1, n), np.random.uniform(0, 1, m)  # random weights

    return xs, ys, p, q
