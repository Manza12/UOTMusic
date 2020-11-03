import os.path as path
import numpy as np

# Synthesis
FS = 44100
DURATION_SYNTHESIS = 4
N = FS * DURATION_SYNTHESIS

# Play audio
MASTER_VOLUME = .1

# Frequencies
REFERENCE_FREQUENCY = 440  # in Hertz
SCALE_NOTES = 12
THRESHOLD_PIPTRACK = 0.5

# Paths
AUDIO_PATH = path.join('data', 'audio')
FIGURES_PATH = path.join('data', 'figures')
RESULTS_PATH = 'results'

# Unbalanced optimal transport
COST_TYPE = "square"
CREATE_FIGURES = True
STRAIGHT = True
LAMBDA = 2e-5
RHO = 1
TOL = 1e-5
THR = 1e-2

# Computation
EPSILON = np.finfo(float).eps
