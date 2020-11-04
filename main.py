""" This is the main script from the project:
Unbalanced Optimal Transport for Sound Synthesis """

import scipy.interpolate as interp
import pylab as plt
from plot import plot_gamma, plot_marginals, plot_interpolations, plot_frequencies, plot_amplitudes
from synthesis import phase_vocoder
from play import play_sound
from parameters import *
from interpolation import create_cost, scaling, music_interp_measures
from piptrack import get_data
from utilities import freq2note, note2freq, create_data
import time
import scipy.io.wavfile as wav
import sys

PLOT_MARGINALS = False
PLOT_FIGURES = False
PLOT_FREQUENCIES = False
PLOT_AMPLITUDES = False
PLOT_GAMMA = False
PLOT_SPECTRUM = False

SAVE_INTERPOLATIONS = True
SAVE_FIGURES = True
SAVE_SOUND = True

PLAY_SOUND = True

if __name__ == '__main__':
    # Create folder
    name = "violin2oboe"
    path_result = path.join(RESULTS_PATH, name)

    # Activate plots
    if RUNNING_SHELL:
        plt.ion()
        plt.show()

    if WRITE_LOG_FILE:
        sys.stdout = open(path.join(path_result, name + '.log'), 'w')

    # Time
    time_start = time.time()

    # Create the time array
    t_synthesis = np.arange(0, N) / FS

    # Source sound
    source_name = 'A4_violin'
    f_source, a_source = get_data(source_name, start=1., duration=1, plot_spectrum=PLOT_SPECTRUM)
    # create_data([440, 660], [0.5, 0.25])

    # Target sound
    target_name = 'A4_oboe'
    f_target, a_target = get_data(target_name, start=1., duration=1, plot_spectrum=PLOT_SPECTRUM)
    # create_data([520, 790], [0.6, 0.15])

    # Convert to notes
    n_source = freq2note(f_source)
    n_target = freq2note(f_target)

    # Create interpolation points
    n_interp = 11
    ts = np.linspace(0, 1, n_interp)

    # Create costs
    cost = create_cost(n_source, n_target)

    # Time
    time_post_init = time.time()
    print("Time to initialize:", round(time_post_init - time_start, 3), "seconds.")

    # Time
    time_pre_scaling = time.time()

    # Scaling
    u, v, gamma, errs = scaling(cost, a_source, a_target, lam=LAMBDA, rho=RHO, tol=TOL)

    # Time
    time_post_scaling = time.time()
    print("Time to compute scaling:", round(time_post_scaling - time_pre_scaling, 3), "seconds.")

    # Plot transport plan
    if PLOT_GAMMA:
        plot_gamma(gamma)

    gamma_a = gamma * np.expand_dims(np.exp(u), 1)
    gamma_b = gamma * np.expand_dims(np.exp(v), 0)

    # Checks that its first marginal is indeed μ (up to numerical error)
    first_marginal = np.sum(gamma_a, axis=1)
    first_marginal_error = np.sum(np.abs(first_marginal / a_source - 1))
    print("First marginal error:", first_marginal_error)

    # Checks that its second marginal is indeed ν (up to optimization error)
    second_marginal = np.sum(gamma_b, axis=0)
    second_marginal_error = np.sum(np.abs(second_marginal / a_target - 1))
    print("Second marginal error:", second_marginal_error)

    if PLOT_MARGINALS:
        plot_marginals(n_source, a_source, n_target, a_target, first_marginal, second_marginal, path_result,
                       save_figures=SAVE_FIGURES)

    # Time
    time_pre_interp = time.time()

    # Interpolation
    a_interp, n_interp = music_interp_measures(ts, n_source, n_target, u, v, gamma)

    # Time
    time_post_interp = time.time()
    print("Time to compute interpolations:", round(time_post_interp - time_pre_interp, 3), "seconds.")

    if PLOT_FIGURES:
        plot_interpolations(n_source, a_source, n_target, a_target, n_interp, a_interp, path_result,
                            save_interpolations=SAVE_INTERPOLATIONS)

    # Create the frequencies and the amplitudes
    interpolator_frequencies = interp.interp1d(ts * DURATION_SYNTHESIS, note2freq(n_interp),
                                               kind='linear', axis=-1, copy=False, assume_sorted=True)
    frequencies_tensor = interpolator_frequencies(t_synthesis)

    interpolator_amplitudes = interp.interp1d(ts * DURATION_SYNTHESIS, a_interp,
                                              kind='linear', axis=-1, copy=False, assume_sorted=True)
    amplitudes_tensor = interpolator_amplitudes(t_synthesis)

    if PLOT_FREQUENCIES:
        plot_frequencies(t_synthesis, frequencies_tensor, path_result, save_figures=SAVE_FIGURES)

    if PLOT_AMPLITUDES:
        plot_amplitudes(t_synthesis, amplitudes_tensor, path_result, save_figures=SAVE_FIGURES)

    # Generate the sound
    y = phase_vocoder(frequencies_tensor, amplitudes_tensor)

    # Play the sound
    if PLAY_SOUND:
        play_sound(y)
    if SAVE_SOUND:
        wav.write(path.join(path_result, name + '.wav'), FS, MASTER_VOLUME * y)
