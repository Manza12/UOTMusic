# This is the main script from the project:
# Unbalanced Optimal Transport for Sound Synthesis

import numpy as np
import scipy.interpolate as interp
from pathlib import Path
from synthesis import phase_vocoder
from play import play_sound
from parameters import *
from interpolation import create_cost, scaling, conic_interp_measures
from utilities import freq2note, note2freq
import pylab as plt
import time
import scipy.io.wavfile as wav
import sys

PLOT_MARGINALS = True
PLOT_FIGURES = True
PLOT_FREQUENCIES = True
PLOT_AMPLITUDES = True

SAVE_INTERPOLATIONS = True
SAVE_FIGURES = True
SAVE_SOUND = True

PLAY_SOUND = True

NOTE_MIN = -5
NOTE_MAX = 17
AMPL_MIN = 0
AMPL_MAX = 1


def plot_figures():
    for k in range(np.size(a_interp, 1)):
        fig = plt.figure(figsize=[6, 4])
        plt.stem(n_source, a_source, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k",
                 use_line_collection=True)
        plt.stem(n_target, a_target, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k",
                 use_line_collection=True)
        plt.stem(n_interp[:, k], a_interp[:, k], linefmt="k", markerfmt="ko", label="\\mu_t", basefmt="k",
                 use_line_collection=True)
        plt.xlim(NOTE_MIN, NOTE_MAX)
        plt.ylim(AMPL_MIN, AMPL_MAX)
        fig.canvas.manager.window.wm_geometry('+500+150')

        if SAVE_INTERPOLATIONS:
            path_to_interpolations = path.join(path_result, 'interpolations')
            Path(path_to_interpolations).mkdir(parents=True, exist_ok=True)
            plt.savefig(path.join(path_to_interpolations, "interp_" + str(k) + ".png"))

    plt.show()


def plot_marginals():
    plt.figure(1, figsize=[6, 4])

    plt.subplot(211)
    plt.stem(n_source, a_source, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k",
             use_line_collection=True)
    plt.stem(n_target, a_target, linefmt="C2", markerfmt="C2o", label="\\nu", basefmt="k",
             use_line_collection=True)
    plt.xlabel("Note")
    plt.ylabel("Amplitude")
    plt.legend(["$\\mu$", "$\\nu$"])
    plt.xlim(NOTE_MIN, NOTE_MAX)
    plt.ylim(AMPL_MIN, AMPL_MAX)
    plt.title("Source and target Dirac's $(\\mu, \\nu)$")

    plt.subplot(212)
    plt.stem(n_source, first_marginal, linefmt="C1", markerfmt="C1o", label="\\mu_0", basefmt="k",
             use_line_collection=True)
    plt.stem(n_target, second_marginal, linefmt="C3", markerfmt="C3o", label="\\nu_0", basefmt="k",
             use_line_collection=True)
    plt.xlabel("Note")
    plt.ylabel("Amplitude")
    plt.legend(["$\\mu_0$", "$\\nu_0$"])
    plt.xlim(NOTE_MIN, NOTE_MAX)
    plt.ylim(AMPL_MIN, AMPL_MAX)
    plt.title("Approximations marginals $(\\mu_0, \\nu_0)$")

    plt.tight_layout()

    if SAVE_FIGURES:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "marginals.png"))

    plt.show()


def plot_frequencies():
    plt.figure()
    plt.plot(t_synthesis, np.transpose(frequencies_tensor))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequencies (Hz)")
    plt.ylim(250, 2500)
    plt.yscale('log')
    plt.title("Frequencies respect to time")
    if SAVE_FIGURES:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "frequencies.png"))
    plt.show()


def plot_amplitudes():
    plt.figure()
    plt.plot(t_synthesis, np.transpose(amplitudes_tensor))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitudes")
    plt.ylim(0, 1)
    # plt.yscale('log')
    plt.title("Amplitudes respect to time")
    if SAVE_FIGURES:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "amplitudes.png"))
    plt.show()


if __name__ == '__main__':
    # Create folder
    name = "do5+do6-la4+la5"
    path_result = path.join(RESULTS_PATH, name)
    Path(path_result).mkdir(parents=True, exist_ok=True)
    sys.stdout = open(path.join(path_result, name + '.log'), 'w')

    # Time
    time_start = time.time()

    # Create the time array
    t_synthesis = np.arange(0, N) / FS

    # Create starting frequencies and amplitudes
    f_source, a_source = np.array([520, 1040]), np.array([0.9, 0.5])
    f_target, a_target = np.array([440, 880]), np.array([0.8, 0.6])

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
    u, v, gamma, errs = scaling(cost, a_source, a_target, lam=1e-2, rho=1, tol=1e-14)

    # Time
    time_post_scaling = time.time()
    print("Time to compute scaling:", round(time_post_scaling - time_pre_scaling, 3), "seconds.")

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
        plot_marginals()

    time_pre_interp = time.time()
    a_interp, n_interp = conic_interp_measures(n_source, n_target, a_source, a_target, ts, lam=1e-2, tol=1e-14,
                                               thr=1e-14)
    # Time
    time_post_interp = time.time()
    print("Time to compute interpolations:", round(time_post_interp - time_pre_interp, 3), "seconds.")

    if PLOT_FIGURES:
        plot_figures()

    # Create the frequencies and the amplitudes
    interpolator_frequencies = interp.interp1d(ts * DURATION_SYNTHESIS, note2freq(n_interp),
                                               kind='linear', axis=-1, copy=False, assume_sorted=True)
    frequencies_tensor = interpolator_frequencies(t_synthesis)

    interpolator_amplitudes = interp.interp1d(ts * DURATION_SYNTHESIS, a_interp,
                                              kind='linear', axis=-1, copy=False, assume_sorted=True)
    amplitudes_tensor = interpolator_amplitudes(t_synthesis)

    if PLOT_FREQUENCIES:
        plot_frequencies()

    if PLOT_AMPLITUDES:
        plot_amplitudes()

    # Generate the sound
    y = phase_vocoder(frequencies_tensor, amplitudes_tensor)

    # Play the sound
    if PLAY_SOUND:
        play_sound(y)
    if SAVE_SOUND:
        wav.write(path.join(path_result, name + '.wav'), FS, MASTER_VOLUME * y)
