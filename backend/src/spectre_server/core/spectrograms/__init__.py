# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create and transform spectrogram data."""

from ._spectrogram import Spectrogram, FrequencyCut, TimeCut, SpectrumUnit, TimeType
from ._array_operations import get_moving_average_window_size
from ._transform import (
    frequency_chop,
    time_chop,
    frequency_average,
    time_average,
    ignore_leading_spectrums,
    join_spectrograms,
)

__all__ = [
    "Spectrogram",
    "FrequencyCut",
    "TimeCut",
    "SpectrumUnit",
    "frequency_chop",
    "time_chop",
    "frequency_average",
    "time_average",
    "ignore_leading_spectrums",
    "join_spectrograms",
    "TimeType",
    "get_moving_average_window_size",
]
