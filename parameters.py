import os.path as p
# Synthesis
fs = 44100
duration_synthesis = 1
N = fs*duration_synthesis

# Play audio
master_volume = 0.2

# Paths
audio_path = p.join('data', 'audio')
