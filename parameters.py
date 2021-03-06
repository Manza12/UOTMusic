import os.path as path
import numpy as np

# Synthesis
FS = 44100
DURATION_SYNTHESIS = 10
N = FS * DURATION_SYNTHESIS

# Play audio
MASTER_VOLUME = 1e0

# Frequencies
REFERENCE_FREQUENCY = 440  # in Hertz
SCALE_NOTES = 1
THRESHOLD_PIPTRACK = 0.1

# Paths
AUDIO_PATH = path.join('data', 'audio')
FIGURES_PATH = path.join('data', 'figures')
RESULTS_PATH = 'results'

# Unbalanced optimal transport
COST_TYPE = "square"
CREATE_FIGURES = True
STRAIGHT = True
LAMBDA = 1e-2
RHO = 1e0
TOL = 1e-6
THR = 1e-6

# Computation
EPSILON = np.finfo(float).eps

# Logs
WRITE_LOG_FILE = False

# Running
RUNNING_SHELL = False
