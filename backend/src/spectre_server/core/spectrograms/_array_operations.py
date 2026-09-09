# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import typing

import numpy as np
import numpy.typing as npt


def moving_average(
    array: npt.NDArray[np.float32], window_size: int, axis: int = 0
) -> npt.NDArray[np.float32]:
    """Averages ``array`` along ``axis`` in non-overlapping windows of ``window_size``.

    Any trailing samples that do not complete a window are discarded.

    :param array: Input array to be averaged.
    :param window_size: Number of items in the window.
    :param axis: Axis along which to perform the averaging, defaults to 0.
    :return: A new array, averaged along the specified axis.
    :raises ValueError: If ``window_size`` is less than one, or greater than the
    length of ``axis``.
    """
    if window_size < 1:
        raise ValueError(
            f"Cannot average over windows of size {window_size}, must be more than one"
        )

    axis_length = array.shape[axis]
    if window_size > axis_length:
        raise ValueError(
            f"The window size ({window_size}) cannot be greater than the length "
            f"of the axis ({axis}). Got axis length {axis_length}."
        )

    num_windows = axis_length // window_size

    slicer = [slice(None)] * array.ndim
    slicer[axis] = slice(0, num_windows * window_size)
    trimmed = array[tuple(slicer)]

    new_shape = list(trimmed.shape)
    new_shape[axis] = num_windows
    new_shape.insert(axis + 1, window_size)
    return np.nanmean(trimmed.reshape(new_shape), axis=axis + 1)


T = typing.TypeVar("T", np.float32, np.datetime64)


def find_closest_index(
    target_value: T, array: npt.NDArray[T], enforce_strict_bounds: bool = False
) -> int:
    """
    Finds the index of the closest value to a target in a given array, with optional bounds enforcement.

    :param target_value: The value to find the closest match for.
    :param array: The array to search within.
    :param enforce_strict_bounds: If True, raises an error if the target value is outside the array bounds. Defaults to False.
    :return: The index of the closest value in the array.
    :raises ValueError: If `enforce_strict_bounds` is True and `target_value` is outside the array bounds.
    """
    # Check bounds if strict enforcement is required
    if enforce_strict_bounds:
        max_value, min_value = np.nanmax(array), np.nanmin(array)
        if target_value > max_value:
            raise ValueError(
                f"Target value {target_value} exceeds max array value {max_value}"
            )
        if target_value < min_value:
            raise ValueError(
                f"Target value {target_value} is less than min array value {min_value}"
            )

    # Find the index of the closest value
    return int(np.argmin(np.abs(array - target_value)))


def normalise_peak_intensity(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Normalises an array by its peak intensity.

    :param array: Input array to normalise.
    :return: Array normalised such that its maximum value is 1. NaN values are ignored.
    """
    return array / np.nanmax(array)


def compute_resolution(array: npt.NDArray[np.float32]) -> float:
    """
    Computes the median resolution of a one-dimensional array.

    :param array: Input one-dimensional array of values.
    :return: The median of differences between consecutive elements in the array.
    :raises ValueError: If the input array is not one-dimensional or contains fewer than two elements.
    """
    # Check that the array is one-dimensional
    if array.ndim != 1:
        raise ValueError("Input array must be one-dimensional")

    if len(array) < 2:
        raise ValueError("Input array must contain at least two elements")

    # Calculate differences between consecutive elements.
    resolutions = np.diff(array)
    return float(np.nanmedian(resolutions))


def compute_range(array: npt.NDArray[np.float32]) -> float:
    """
    Computes the range of a one-dimensional array as the difference between its last and first elements.

    :param array: Input one-dimensional array of values.
    :return: The range of the array (last element minus first element).
    :raises ValueError: If the input array is not one-dimensional or contains fewer than two elements.
    """
    # Check that the array is one-dimensional
    if array.ndim != 1:
        raise ValueError("Input array must be one-dimensional")

    if len(array) < 2:
        raise ValueError("Input array must contain at least two elements")
    return float(array[-1] - array[0])


def subtract_background(
    array: npt.NDArray[np.float32], start_index: int, end_index: int
) -> npt.NDArray[np.float32]:
    """
    Subtracts the mean of a specified background range from all elements in an array.

    :param array: Input array from which the background mean will be subtracted.
    :param start_index: Start index of the background range (inclusive).
    :param end_index: End index of the background range (inclusive).
    :return: Array with the background mean subtracted.
    """
    array -= np.nanmean(array[start_index : end_index + 1])
    return array


def time_elapsed(datetimes: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float32]:
    """Convert an array of datetimes to seconds elapsed."""
    return (datetimes - datetimes[0]).astype("timedelta64[us]") / np.timedelta64(1, "s")
