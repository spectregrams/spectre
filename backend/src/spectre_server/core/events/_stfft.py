# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import typing

import numpy as np
import numpy.typing as npt
import pyfftw

import spectre_server.core.fields


def get_cosine_signal(
    num_samples: int,
    sample_rate: float,
    frequency: float,
    amplitude: float,
    phase: float,
) -> npt.NDArray[np.complex64]:
    """Generate a complex-valued cosine wave.

    :param num_samples: The number of samples in the signal.
    :param sample_rate: The rate at which the signal is sampled.
    :param frequency: The frequency of the cosine.
    :param amplitude: The amplitude of the cosine.
    :param phase: The initial phase of the cosine.
    :return: A numpy array containing the complex-valued signal.
    """
    n = np.arange(num_samples, dtype=np.complex64)
    return amplitude * np.cos(2 * np.pi * (frequency / sample_rate) * n + phase)


def _window_general_cosine_asym(
    window_size: int,
    coefficients: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    angles = np.linspace(-np.pi, np.pi, window_size + 1)[:-1]
    window = sum(coef * np.cos(k * angles) for k, coef in enumerate(coefficients))
    return window.astype(np.float32)


def _window_boxcar(window_size: int) -> npt.NDArray[np.float32]:
    return np.ones(window_size, np.float32)


def _window_hann(window_size: int) -> npt.NDArray[np.float32]:
    coefficients = np.asarray([0.5, 0.5])
    return _window_general_cosine_asym(window_size, coefficients)


def _window_blackman(window_size: int) -> npt.NDArray[np.float32]:
    coefficients = np.asarray([0.42, 0.50, 0.08])
    return _window_general_cosine_asym(window_size, coefficients)


def get_window(window_type: str, window_size: int) -> npt.NDArray[np.float32]:
    """Create a window of a specified type and length.

    :param window_type: The type of window to generate.
    :param window_size: The number of samples in the window.
    :return: A numpy array containing the window samples.
    :raises ValueError: If window_size is less than two or an unknown window type is provided.
    """
    if window_size < 2:
        raise ValueError(f"The window size cannot be less than 2, got {window_size}")

    if window_type == spectre_server.core.fields.WindowType.BOXCAR:
        return _window_boxcar(window_size)
    elif window_type == spectre_server.core.fields.WindowType.HANN:
        return _window_hann(window_size)
    elif window_type == spectre_server.core.fields.WindowType.BLACKMAN:
        return _window_blackman(window_size)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def get_buffer(num_samples: int) -> npt.NDArray[np.complex64]:
    """Create an empty, memory-aligned buffer for in-place DFTs carried out by FFTW.

    :param num_samples: The number of samples in the buffer.
    :return: An empty numpy array.
    """
    return pyfftw.empty_aligned(num_samples, dtype="complex64")


def get_fftw_obj(buffer: npt.NDArray[np.complex64]) -> pyfftw.FFTW:
    """Plan an in-place 1D DFT using FFTW.

    The contents of the input buffer will be overwritten during the planning process, and so
    should be initialised after this function is called.

    :param buffer: An empty numpy array.
    :return: An FFTW object that, when called, computes the forward FFT of whatever is in the buffer.
    """
    return pyfftw.FFTW(buffer, buffer, direction="FFTW_FORWARD", flags=["FFTW_PATIENT"])


def get_times(
    num_spectrums: int, sample_rate: float, window_hop: int
) -> npt.NDArray[np.float32]:
    """Get the physical times assigned to each spectrum in a spectrogram.

    The first spectrum is by convention at t=0. Each spectrum thereafter is offset by `window_hop` samples,
    corresponding to `window_hop * sample_interval` units, where `sample_interval` is the inverse of the
    sample rate.

    :param num_spectrums: The number of spectrums in the spectrogram.
    :param sample_rate: The sample rate of the signal.
    :param window_hop: The number of samples the window shifts per frame.
    :return: A numpy array containing the times assigned to each spectrum.
    """
    return np.arange(num_spectrums, dtype=np.float32) * window_hop * 1.0 / sample_rate


def get_frequencies(window_size: int, sample_rate: float) -> npt.NDArray[np.float32]:
    """Compute the DFT sample frequencies. for a given window size and sample rate.

    :param window_size: The number of samples in each window.
    :param sample_rate: The sample rate at which the signal was captured.
    :return: A numpy array containing the sample frequencies, with zero at the start.
    """
    return np.fft.fftfreq(window_size, d=np.float32(1.0 / sample_rate))


def get_num_spectrums(signal_size: int, window_size: int, window_hop: int) -> int:
    """Compute the number of spectrums in the spectrogram.

    The first window is centered at the start of the signal (index 0). The last window is the final one that
    fits entirely within the signal.

    :param signal_size: The total number of samples in the signal.
    :param window_size: The number of samples in each window.
    :param window_hop: The number of samples the window is shifted in each frame.
    :return: The total number of spectrums in the resulting spectrogram, when performing an
    stfft with these values.
    :raises ValueError: If the window size or hop is less than one sample, or the window size is greater than the signal size.
    """
    if window_size < 1:
        raise ValueError(
            f"The window size must be at least one. " f"Got {window_size}."
        )

    if window_hop < 1:
        raise ValueError(f"The window hop must be at least one. " f"Got {window_hop}.")

    if window_size > signal_size:
        raise ValueError(
            f"The window must fit within the signal. "
            f"Got window size {window_size}, which is greater "
            f"than the signal size {signal_size}."
        )

    return int((signal_size - np.ceil(window_size / 2)) / window_hop) + 1


def get_num_dangling_samples(window_size: int) -> int:
    """Compute the number of samples by which the first window dangles.

    By convention, the first window is centered at the start of the signal (index 0).

    :param window_size: The number of samples in each window.
    :return: The number of samples by which the first window dangles.
    """
    return window_size // 2


def stfft(
    fftw_obj: pyfftw.FFTW,
    buffer: npt.NDArray[np.complex64],
    signal: npt.NDArray[np.complex64],
    window: npt.NDArray[np.float32],
    window_hop: int,
    prepend_signal: typing.Optional[npt.NDArray[np.complex64]] = None,
) -> npt.NDArray[np.float32]:
    """Compute the short-time discrete Fourier transform of the input signal, using a real sliding window.

    The first window is centered at the start of the signal (index 0). The last window is the final one that
    fits entirely within the signal.

    :param fftw_obj: An FFTW object, pre-planned for in-place transforms on the buffer.
    :param buffer: An empty numpy array, used for repeated in-place DFTs.
    :param signal: The input signal.
    :param window: The window function, same length as the buffer.
    :param window_hop: The number of samples the window advances per frame.
    :param prepend_signal: The value assumed by the signal where the window dangles. If not provided (default),
    by convention those samples are zeroed.
    :return: a spectrogram containing the amplitude of each spectral component.
    :raises ValueError: If the window and buffer sizes do not match, or if `prepend_signal` is provided
    with a length other than `window_size // 2`.
    """
    window_size = window.shape[0]
    buffer_size = buffer.shape[0]
    signal_size = signal.shape[0]

    if window_size != buffer_size:
        raise ValueError(
            f"The window and buffer must be the same size. "
            f"Got that the window has {window_size} samples, "
            f"but the buffer has {buffer_size} samples."
        )

    # Calculate how many spectrums will be in the spectrogram.
    num_spectrums = get_num_spectrums(signal_size, window_size, window_hop)

    # Initialise an empty array, into which we'll copy the spectrums computed by fftw.
    dynamic_spectra = np.empty((window_size, num_spectrums), dtype=np.float32)

    num_dangling_samples = get_num_dangling_samples(window_size)
    if prepend_signal is None:
        prepend_signal = np.zeros((num_dangling_samples,), dtype=np.complex64)
    elif prepend_signal.shape[0] != num_dangling_samples:
        raise ValueError(
            f"Must provide exactly {num_dangling_samples} samples to prepend, "
            f"got {prepend_signal.shape[0]}."
        )

    for n in range(num_spectrums):
        # Center the window for the current frame
        center = window_hop * n
        start = center - window_size // 2
        stop = start + window_size

        # The window is fully inside the signal.
        if start >= 0 and stop <= signal_size:
            buffer[:] = signal[start:stop] * window

        # The window partially overlaps with the signal.
        else:
            signal_indices = np.arange(start, stop)

            out_mask = signal_indices < 0
            buffer[out_mask] = prepend_signal[
                signal_indices[out_mask] + num_dangling_samples
            ]

            in_mask = ~out_mask
            buffer[in_mask] = signal[signal_indices[in_mask]]

            buffer *= window

        # Compute the DFT in-place, to produce the spectrum.
        fftw_obj.execute()

        # Copy the spectrum into the spectrogram.
        dynamic_spectra[:, n] = np.abs(buffer)

    return dynamic_spectra
