import sounddevice as sd
from parameters import *


def play_sound(y):
    sd.play(master_volume * y, fs)
    status = sd.wait()

    return status
