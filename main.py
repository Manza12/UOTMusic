# This is the main script from the project:
# Unbalanced Optimal Transport for Sound Synthesis
import sounddevice as sd
from synthesis import additive_synthesis
from parameters import *

if __name__ == '__main__':
    frequencies = [440, 660]
    amplitudes = [1, 0.5]

    y = additive_synthesis(frequencies, amplitudes)

    sd.play(master_volume * y, fs)
    status = sd.wait()
