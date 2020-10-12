import numpy as np
from parameters import *


def additive_synthesis(frequencies, amplitudes):
    # Additive synthesis
    t = np.arange(0, N) / fs
    phases = 2 * np.pi * np.expand_dims(frequencies, 1) * np.expand_dims(t, 0)
    amplitudes_time = np.expand_dims(amplitudes, 1) * np.ones(N)
    components = amplitudes_time * np.sin(phases)
    y = np.sum(components, 0)

    return y
