# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import math

import numpy as np

from ._array_operations import find_closest_index, moving_average, time_elapsed
from ._spectrogram import Spectrogram


def frequency_chop(
    spectrogram: Spectrogram, start_frequency: float, end_frequency: float
) -> Spectrogram:
    """
    Extracts a portion of the spectrogram within the specified frequency range.

    :param spectrogram: The input spectrogram to process.
    :param start_frequency: The starting frequency of the desired range (Hz).
    :param end_frequency: The ending frequency of the desired range (Hz).
    :raises ValueError: If the specified frequency range is entirely outside the spectrogram's frequency range.
    :raises ValueError: If the start and end indices for the frequency range are identical.
    :return: A new spectrogram containing only the specified frequency range.
    """
    is_entirely_below_frequency_range = (
        start_frequency < spectrogram.frequencies[0]
        and end_frequency < spectrogram.frequencies[0]
    )
    is_entirely_above_frequency_range = (
        start_frequency > spectrogram.frequencies[-1]
        and end_frequency > spectrogram.frequencies[-1]
    )
    if is_entirely_below_frequency_range or is_entirely_above_frequency_range:
        raise ValueError(
            f"The requested frequency interval is entirely out of range of the input spectrogram."
        )

    # find the index of the nearest matching frequency bins in the spectrogram
    start_index = find_closest_index(
        np.float32(start_frequency), spectrogram.frequencies
    )
    end_index = find_closest_index(np.float32(end_frequency), spectrogram.frequencies)

    # enforce distinct start and end indices
    if start_index == end_index:
        raise ValueError(
            f"Start and end indices are equal! Got start_index: {start_index} and end_index: {end_index}"
        )

    # if start index is more than end index, swap the ordering so to enforce start_index <= end_index
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    # chop the spectrogram accordingly
    transformed_dynamic_spectra = spectrogram.dynamic_spectra[
        start_index : end_index + 1, :
    ]
    transformed_frequencies = spectrogram.frequencies[start_index : end_index + 1]

    return Spectrogram(
        transformed_dynamic_spectra,
        spectrogram.times,
        transformed_frequencies,
        spectrogram.spectrum_unit,
        spectrogram.start_datetime,
    )


def time_chop(
    spectrogram: Spectrogram,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
) -> Spectrogram:
    """
    Extracts a portion of the spectrogram within the specified time range.

    :param spectrogram: The input spectrogram to process.
    :param start_datetime: The starting time of the desired range.
    :param end_datetime: The ending time of the desired range.
    :raises ValueError: If the specified time range is entirely outside the spectrogram's time range.
    :raises ValueError: If the start and end indices for the time range are identical.
    :return: A new spectrogram containing only the specified time range.
    """
    start_datetime64 = np.datetime64(start_datetime)
    end_datetime64 = np.datetime64(end_datetime)

    is_entirely_below_time_range = (
        start_datetime64 < spectrogram.datetimes[0]
        and end_datetime64 < spectrogram.datetimes[0]
    )
    is_entirely_above_time_range = (
        start_datetime64 > spectrogram.datetimes[-1]
        and end_datetime64 > spectrogram.datetimes[-1]
    )
    if is_entirely_below_time_range or is_entirely_above_time_range:
        raise ValueError(
            f"Requested time interval is entirely out of range of the input spectrogram."
        )

    # find the index of the nearest matching spectrums in the spectrogram.
    start_index = find_closest_index(start_datetime64, spectrogram.datetimes)
    end_index = find_closest_index(end_datetime64, spectrogram.datetimes)

    # enforce distinct start and end indices
    if start_index == end_index:
        raise ValueError(
            f"Start and end indices are equal! Got start_index: {start_index} and end_index: {end_index}"
        )

    # if start index is more than end index, swap the ordering so to enforce start_index <= end_index
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    # chop the spectrogram accordingly
    transformed_dynamic_spectra = spectrogram.dynamic_spectra[
        :, start_index : end_index + 1
    ]
    transformed_start_datetime = spectrogram.datetimes[start_index]

    # chop the times array and translate such that the first spectrum to t=0 [s]
    transformed_times = spectrogram.times[start_index : end_index + 1]
    transformed_times -= transformed_times[0]

    return Spectrogram(
        transformed_dynamic_spectra,
        transformed_times,
        spectrogram.frequencies,
        spectrogram.spectrum_unit,
        transformed_start_datetime,
    )


def time_average(spectrogram: Spectrogram, resolution: float) -> Spectrogram:
    """Average a spectrogram in time to a desired resolution by applying a moving average.

    :param spectrogram: The input spectrogram to process.
    :param resolution: The desired time resolution.
    """

    if resolution < spectrogram.time_resolution:
        raise ValueError(
            f"Desired time resolution {resolution} is less than the current {spectrogram.time_resolution}"
        )

    if resolution >= spectrogram.time_range:
        raise ValueError(
            f"Desired time resolution {resolution} must be less than the time range {spectrogram.time_range}"
        )

    window_size = math.floor(resolution / spectrogram.time_resolution)
    transformed_dynamic_spectra = moving_average(
        spectrogram.dynamic_spectra, window_size, axis=1
    )

    # Take the start time of each window. ``moving_average`` drops any trailing
    # partial window, so slice ``times`` to match.
    num_windows = transformed_dynamic_spectra.shape[1]
    transformed_times = spectrogram.times[: num_windows * window_size : window_size]

    return Spectrogram(
        transformed_dynamic_spectra,
        transformed_times,
        spectrogram.frequencies,
        spectrogram.spectrum_unit,
        start_datetime=(
            spectrogram.start_datetime if spectrogram.start_datetime_is_set else None
        ),
    )


def frequency_average(spectrogram: Spectrogram, resolution: float) -> Spectrogram:
    """Average a spectrogram in frequency to a desired resolution by applying a moving average.

    :param spectrogram: The input spectrogram to process.
    :param resolution: The desired frequency resolution.
    """

    if resolution < spectrogram.frequency_resolution:
        raise ValueError(
            f"Desired frequency resolution {resolution} is less than the current {spectrogram.frequency_resolution}"
        )

    if resolution >= spectrogram.frequency_range:
        raise ValueError(
            f"Desired frequency resolution {resolution} must be less than the frequency range {spectrogram.time_range}"
        )

    window_size = math.floor(resolution / spectrogram.frequency_resolution)
    transformed_dynamic_spectra = moving_average(
        spectrogram.dynamic_spectra, window_size, axis=0
    )

    # Contrary to the time average, average over the frequencies corresponding to each spectral component per window.
    transformed_frequencies = moving_average(spectrogram.frequencies, window_size)

    return Spectrogram(
        transformed_dynamic_spectra,
        spectrogram.times,
        transformed_frequencies,
        spectrogram.spectrum_unit,
        start_datetime=(
            spectrogram.start_datetime if spectrogram.start_datetime_is_set else None
        ),
    )


def join_spectrograms(spectrograms: list[Spectrogram]) -> Spectrogram:
    """
    Joins multiple spectrograms into a single spectrogram along the time axis.

    :param spectrograms: A list of spectrograms to combine.
    :raises ValueError: If the input list is empty.
    :raises ValueError: If spectrograms have mismatched frequency ranges.
    :raises ValueError: If spectrograms have different tags.
    :raises ValueError: If spectrograms have differing spectrum units.
    :raises ValueError: If any spectrogram lacks a defined start datetime.
    :return: A new spectrogram combining all input spectrograms along the time axis.
    """
    # check that the length of the list is non-zero
    num_spectrograms = len(spectrograms)
    if num_spectrograms == 0:
        raise ValueError(f"Input list of spectrograms is empty!")

    # extract the first element of the list, and use this as a reference for comparison
    # input validations.
    reference_spectrogram = spectrograms[0]

    # perform checks on each spectrogram in teh list
    for spectrogram in spectrograms:
        if not np.all(
            np.equal(spectrogram.frequencies, reference_spectrogram.frequencies)
        ):
            raise ValueError(f"All spectrograms must have identical frequency ranges")
        if spectrogram.spectrum_unit != reference_spectrogram.spectrum_unit:
            raise ValueError(
                f"All units must be equal for each spectrogram in the input list!"
            )
        if not spectrogram.start_datetime_is_set:
            raise ValueError(f"All spectrograms must have their start datetime set.")

    # Concatenate all dynamic spectra directly along the time axis
    transformed_dynamic_spectra = np.hstack(
        [spectrogram.dynamic_spectra for spectrogram in spectrograms]
    )

    transformed_times = time_elapsed(
        np.concatenate([s.datetimes for s in spectrograms])
    )

    return Spectrogram(
        transformed_dynamic_spectra,
        transformed_times,
        reference_spectrogram.frequencies,
        reference_spectrogram.spectrum_unit,
        reference_spectrogram.start_datetime,
    )
