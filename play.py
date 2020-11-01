import sounddevice as sd
from parameters import *


def play_sound(y):
    sd.play(MASTER_VOLUME * y, FS)
    status = sd.wait()

    return status
