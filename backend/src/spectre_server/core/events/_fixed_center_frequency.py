# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import typing

import numpy as np

import spectre_server.core.batches
import spectre_server.core.spectrograms
import spectre_server.core.fields

from ._base import Base, BaseModel
from ._stfft import (
    get_buffer,
    get_window,
    get_times,
    get_num_spectrums,
    get_num_dangling_windows,
    get_frequencies,
    get_fftw_obj,
    stfft,
)

_LOGGER = logging.getLogger(__name__)


class FixedCenterFrequencyModel(BaseModel):
    window_size: spectre_server.core.fields.Field.window_size = 1024
    window_hop: spectre_server.core.fields.Field.window_hop = 1024
    window_type: spectre_server.core.fields.Field.window_type = (
        spectre_server.core.fields.WindowType.BLACKMAN
    )
    center_frequency: spectre_server.core.fields.Field.center_frequency = 95.8e6
    sample_rate: spectre_server.core.fields.Field.sample_rate = 32e3
    frequency_resolution: spectre_server.core.fields.Field.frequency_resolution = 0
    time_resolution: spectre_server.core.fields.Field.time_resolution = 0
    batch_size: spectre_server.core.fields.Field.batch_size = 3
    keep_signal: spectre_server.core.fields.Field.keep_signal = True
    output_type: spectre_server.core.fields.Field.output_type = (
        spectre_server.core.fields.OutputType.FC32
    )


class FixedCenterFrequency(
    Base[FixedCenterFrequencyModel, spectre_server.core.batches.IQStreamBatch]
):
    def __init__(
        self,
        tag: str,
        model: FixedCenterFrequencyModel,
        batch_cls: typing.Type[spectre_server.core.batches.IQStreamBatch],
    ) -> None:
        super().__init__(tag, model, batch_cls)
        self.__model = model

        # Make the window.
        self.__window = get_window(self.__model.window_type, self.__model.window_size)

        # Pre-allocate the buffer.
        self.__buffer = get_buffer(self.__model.window_size)

        # Defer the expensive FFTW plan creation until the first batch is being processed.
        # With this approach, we avoid a bug where filesystem events are missed because
        # the watchdog observer isn't set up in time before the receiver starts capturing data.
        self.__fftw_obj = None

        self.__output_type = self.__model.output_type

    @property
    def _watch_extension(self) -> str:
        return self.__output_type

    def process(
        self, batch: spectre_server.core.batches.IQStreamBatch
    ) -> spectre_server.core.spectrograms.Spectrogram:
        """Compute the spectrogram of IQ samples captured at a fixed center frequency."""
        _LOGGER.info(f"Reading the I/Q samples")
        iq_data = batch.read_iq(self.__output_type)

        if self.__fftw_obj is None:
            _LOGGER.info(f"Creating the FFTW plan")
            self.__fftw_obj = get_fftw_obj(self.__buffer)

        _LOGGER.info("Executing the short-time FFT")
        dynamic_spectra = stfft(
            self.__fftw_obj,
            self.__buffer,
            iq_data,
            self.__window,
            self.__model.window_hop,
        )

        # Compute the physical times we'll assign to each spectrum.
        num_spectrums = get_num_spectrums(
            iq_data.size, self.__model.window_size, self.__model.window_hop
        )
        times = get_times(
            num_spectrums, self.__model.sample_rate, self.__model.window_hop
        )

        # Get the physical frequencies assigned to each spectral component, shift the zero frequency to the middle of the
        # spectrum, then translate the array up from the baseband.
        frequencies = (
            np.fft.fftshift(
                get_frequencies(self.__model.window_size, self.__model.sample_rate)
            )
            + self.__model.center_frequency
        )

        # Shift the zero-frequency component to the middle of the spectrum.
        dynamic_spectra = np.fft.fftshift(dynamic_spectra, axes=0)

        _LOGGER.info("Creating the spectrogram")
        spectrogram = spectre_server.core.spectrograms.Spectrogram(
            dynamic_spectra,
            times,
            frequencies,
            spectre_server.core.spectrograms.SpectrumUnit.AMPLITUDE,
            batch.start_datetime,
        )

        target_time_resolution = max(
            self.__model.time_resolution, spectrogram.time_resolution
        )

        # Only ignore leading spectrums if they will be averaged away. We don't want NaN values in the final spectrogram.
        num_dangling_windows = get_num_dangling_windows(
            self.__model.window_size, self.__model.window_hop
        )
        moving_average_window_size = (
            spectre_server.core.spectrograms.get_moving_average_window_size(
                target_time_resolution, spectrogram.time_resolution
            )
        )
        will_average = moving_average_window_size > 1
        nans_averaged_away = num_dangling_windows < moving_average_window_size
        if will_average and nans_averaged_away:
            spectrogram = spectre_server.core.spectrograms.ignore_leading_spectrums(
                spectrogram, num_dangling_windows
            )

        spectrogram = spectre_server.core.spectrograms.time_average(
            spectrogram, target_time_resolution
        )
        spectrogram = spectre_server.core.spectrograms.frequency_average(
            spectrogram,
            max(self.__model.frequency_resolution, spectrogram.frequency_resolution),
        )

        _LOGGER.info("Spectrogram created successfully")

        if not self.__model.keep_signal:
            _LOGGER.info(f"Deleting the I/Q samples")
            batch.delete_iq(self.__output_type)

        return spectrogram
