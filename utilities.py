from pathlib import Path
import numpy as np


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
