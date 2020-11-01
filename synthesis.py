from piptrack import get_data
from parameters import *
from play import play_sound


def phase_vocoder(frequencies_tensor, amplitudes_tensor):
    """Create a sound from time variable frequencies and amplitudes.

        This function creates a sound from time variable frequencies and amplitudes.

        Parameters
        ----------
        frequencies_tensor : array_like
            Frequencies along time. The rows are channels of varying frequencies.
        amplitudes_tensor : array_like
            Amplitudes along time. The rows are channels of varying amplitudes.
        """

    # Check that the both tensors have the same size
    assert (np.size(frequencies_tensor) == np.size(amplitudes_tensor))

    # Check that the second dimension of the frequencies tensor is the same as the time array
    assert(np.size(frequencies_tensor, 1) == N) ## Here it is actually not true, we have a lot fewer actual frequencies that N, no? Moreover the frequency tensor is only a vector (uni-dimensional)

    # Create the phases tensor
    phases = 2 * np.pi * np.cumsum(frequencies_tensor / FS, 1)

    # Create the components tensor
    components = amplitudes_tensor * np.sin(phases)

    # Sum over components
    y = np.sum(components, 0)

    return y


def additive_synthesis(frequencies, amplitudes):
    """ Create a sound from fixed frequencies and amplitudes by additive synthesis.

        This function creates a sound from fixed frequencies and amplitudes.

        Parameters
        ----------
        frequencies : array_like
            Array of frequencies that form the signal.
        amplitudes : array_like
            Array of amplitudes that form the signal
    """

    # Create the time array
    t = np.arange(0, N) / FS

    # Create the phases tensor
    phases = 2 * np.pi * np.expand_dims(frequencies, 1) * np.expand_dims(t, 0)

    # Create the amplitudes tensor
    amplitudes_tensor = np.expand_dims(amplitudes, 1) * np.ones(N)

    # Create the components tensor
    components = amplitudes_tensor * np.sin(phases)

    # Sum over components
    y = np.sum(components, 0)

    return y


if __name__ == '__main__':
    name = 'lam'
    _frequencies, _amplitudes = get_data(name, start=0.01, duration=1.)

    _y = additive_synthesis(_frequencies, _amplitudes)

    play_sound(_y)
