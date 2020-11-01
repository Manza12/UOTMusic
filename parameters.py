import os.path as path
# Synthesis
fs = 44100
duration_synthesis = 1
N = fs*duration_synthesis

# Play audio
master_volume = 0.2

# Paths
audio_path = path.join('data', 'audio')
figures_path = path.join('data', 'figures')

# Unbalanced optimal transport
cost_type = "conic"
create_figures = False
