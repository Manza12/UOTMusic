import os.path as path
# Synthesis
FS = 44100
DURATION_SYNTHESIS = 4
N = FS * DURATION_SYNTHESIS

# Play audio
MASTER_VOLUME = 0.2

# Frequencies
REFERENCE_FREQUENCY = 440  # in Hertz
SCALE_NOTES = 12

# Paths
AUDIO_PATH = path.join('data', 'audio')
FIGURES_PATH = path.join('data', 'figures')
RESULTS_PATH = 'results'

# Unbalanced optimal transport
COST_TYPE = "square"
CREATE_FIGURES = True
STRAIGHT = True
