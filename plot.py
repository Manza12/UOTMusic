import pylab as plt
from parameters import *
from utilities import note2freq
from pathlib import Path
from parameters import RUNNING_SHELL

AMPL_MIN = 1e-4
AMPL_MAX = 1

AMPL_LOG = True


def plot_interpolations(n_source, a_source, n_target, a_target, n_interp, a_interp, path_result,
                        save_interpolations=True):
    freq_source = note2freq(n_source)
    freq_target = note2freq(n_target)

    min_freq = min(np.min(freq_source), np.min(freq_target))
    max_freq = max(np.max(freq_source), np.max(freq_target))

    for k in range(np.size(a_interp, 1)):
        fig = plt.figure(figsize=[6, 4])

        freq_interp = note2freq(n_interp[:, k])

        plt.stem(freq_source, a_source, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k",
                 use_line_collection=True)
        plt.stem(freq_target, a_target, linefmt="C3", markerfmt="C3o", label="\\nu", basefmt="k",
                 use_line_collection=True)
        plt.stem(freq_interp, a_interp[:, k], linefmt="k", markerfmt="ko", label="\\mu_t", basefmt="k",
                 use_line_collection=True)

        plt.xlim(min_freq*0.9, max_freq*1.1)
        plt.xscale('log')

        plt.ylim(AMPL_MIN, AMPL_MAX)
        if AMPL_LOG:
            plt.yscale('log')

        if not RUNNING_SHELL:
            fig.canvas.manager.window.wm_geometry('+500+150')

        if save_interpolations:
            path_to_interpolations = path.join(path_result, 'interpolations')
            Path(path_to_interpolations).mkdir(parents=True, exist_ok=True)
            plt.savefig(path.join(path_to_interpolations, "interp_" + str(k) + ".png"))

    plt.show()


def plot_marginals(n_source, a_source, n_target, a_target, first_marginal, second_marginal, path_result,
                   save_figures=True):
    plt.figure(1, figsize=[6, 4])
    plt.subplot(211)

    freq_source = note2freq(n_source)
    plt.stem(freq_source, a_source, linefmt="C0", markerfmt="C0o", label="\\mu", basefmt="k",
             use_line_collection=True)

    freq_target = note2freq(n_target)
    plt.stem(freq_target, a_target, linefmt="C2", markerfmt="C2o", label="\\nu", basefmt="k",
             use_line_collection=True)

    min_freq = min(np.min(freq_source), np.min(freq_target))
    max_freq = max(np.max(freq_source), np.max(freq_target))

    plt.xlabel("Note")
    plt.xlim(min_freq, max_freq)
    plt.xscale('log')

    plt.ylabel("Amplitude")

    plt.legend(["$\\mu$", "$\\nu$"])

    if AMPL_LOG:
        plt.ylim(AMPL_MIN, AMPL_MAX)
        plt.yscale('log')
    else:
        plt.ylim(0, 1)

    plt.title("Source and target Dirac's $(\\mu, \\nu)$")

    plt.subplot(212)
    plt.stem(freq_source, first_marginal, linefmt="C1", markerfmt="C1o", label="\\mu_0", basefmt="k",
             use_line_collection=True)
    plt.stem(freq_target, second_marginal, linefmt="C3", markerfmt="C3o", label="\\nu_0", basefmt="k",
             use_line_collection=True)

    plt.xlabel("Note")
    plt.ylabel("Amplitude")
    plt.legend(["$\\mu_0$", "$\\nu_0$"])

    plt.xlim(min_freq, max_freq)
    plt.xscale('log')

    plt.ylim(AMPL_MIN, AMPL_MAX)
    if AMPL_LOG:
        plt.yscale('log')

    plt.title("Approximations marginals $(\\mu_0, \\nu_0)$")

    plt.tight_layout()

    if save_figures:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "marginals.png"))

    plt.show()


def plot_frequencies(t_synthesis, frequencies_tensor, path_result, save_figures=True):
    plt.figure()
    plt.plot(t_synthesis, np.transpose(frequencies_tensor))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequencies (Hz)")
    plt.ylim(250, 2500)
    plt.yscale('log')
    plt.title("Frequencies respect to time")
    if save_figures:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "frequencies.png"))
    plt.show()


def plot_amplitudes(t_synthesis, amplitudes_tensor, path_result, save_figures=True):
    plt.figure()
    plt.plot(t_synthesis, np.transpose(amplitudes_tensor))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitudes")
    plt.ylim(0, 1)
    if AMPL_LOG:
        plt.ylim(1e-4, 1)
        plt.yscale('log')

    plt.title("Amplitudes respect to time")
    if save_figures:
        path_to_figures = path.join(path_result, 'figures')
        Path(path_to_figures).mkdir(parents=True, exist_ok=True)
        plt.savefig(path.join(path_to_figures, "amplitudes.png"))
    plt.show()


def plot_gamma(transport_plan, log_scale=AMPL_LOG):
    plt.figure()

    if log_scale:
        z = np.log(np.transpose(transport_plan) + EPSILON)
        plt.imshow(z, vmin=-7, cmap='Greys', origin='lower')
    else:
        z = np.transpose(transport_plan)
        plt.imshow(z, vmin=0, vmax=np.max(z), cmap='Greys', origin='lower')

    plt.title("Transport plan")

    plt.show()


def plot_transport_plan(x, y, transport_plan, log_scale=False):
    plt.figure()

    if log_scale:
        z = np.log(np.transpose(transport_plan) + EPSILON)
        plt.pcolor(x, y, z, vmin=-7, cmap='Greys')
    else:
        z = np.transpose(transport_plan)
        plt.pcolor(x, y, z, vmin=0, vmax=1, cmap='Greys')

    plt.title("Transport plan")

    plt.show()
