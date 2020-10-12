# This is the main script from the project:
# Unbalanced Optimal Transport for Sound Synthesis

import numpy as np
from synthesis import phase_vocoder
from play import play_sound
from parameters import *

if __name__ == '__main__':
    # Create the time array
    t = np.arange(0, N) / fs

    # Create the frequencies and the amplitudes
    frequencies_tensor = np.stack((np.interp(t, [0, 1], [440, 880]), np.interp(t, [0, 1], [660, 1320])))
    amplitudes_tensor = np.stack((np.ones(N), np.ones(N)))

    # Generate the sound
    y = phase_vocoder(frequencies_tensor, amplitudes_tensor)

    # Play the sound
    play_sound(y)
