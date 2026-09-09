# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import typing

import numpy as np
import numpy.typing as npt

import spectre_server.core.fields
import spectre_server.core.batches

from ._base import Base, BaseModel
from ._stfft import (
    get_buffer,
    get_window,
    get_times,
    get_num_spectrums,
    get_num_dangling_samples,
    get_frequencies,
    get_fftw_obj,
    stfft,
)

_LOGGER = logging.getLogger(__name__)


class CallistoModel(BaseModel):
    window_size: spectre_server.core.fields.Field.window_size = 1024
    window_hop: spectre_server.core.fields.Field.window_hop = 1024
    window_type: spectre_server.core.fields.Field.window_type = (
        spectre_server.core.fields.WindowType.BLACKMAN
    )
    center_frequency: spectre_server.core.fields.Field.center_frequency = 95.8e6
    sample_rate: spectre_server.core.fields.Field.sample_rate = 32e3
    frequency_resolution: spectre_server.core.fields.Field.frequency_resolution = 50000
    time_resolution: spectre_server.core.fields.Field.time_resolution = 0.1
    batch_size: spectre_server.core.fields.Field.batch_size = 3
    keep_signal: spectre_server.core.fields.Field.keep_signal = False
    output_type: spectre_server.core.fields.Field.output_type = (
        spectre_server.core.fields.OutputType.FC32
    )
    # Default value is provided by Alexandro Rabadán Parra, see https://github.com/AlexandroRP99/e-Callisto_Py_RX-888_MK_II
    callisto_scale_factor: spectre_server.core.fields.Field.callisto_scale_factor = (
        89958.629068
    )


def linear_calibration(
    dynamic_spectra: npt.NDArray[np.float32], gradient: float, const: float = 0
) -> npt.NDArray[np.float32]:
    """Apply a calibration relating DFT amplitudes to linearised CALLISTO digits.

    This assumes (of course) that the relationship is linear, and also (more subtly)
    that the relationship is independent of frequency.
    """
    return dynamic_spectra * gradient + const


class Callisto(Base[CallistoModel, spectre_server.core.batches.CallistoBatch]):
    def __init__(
        self,
        tag: str,
        model: CallistoModel,
        batch_cls: typing.Type[spectre_server.core.batches.CallistoBatch],
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

        # Cache the last samples from the previous batch so we can eliminate window effects at the start of
        # the current one. The first batch has no previous, so (by convention) assume the samples are zero.
        self.__prepend_signal = np.zeros(
            get_num_dangling_samples(self.__model.window_size), dtype=np.complex64
        )

    @property
    def _watch_extension(self) -> str:
        return self.__model.output_type

    def process(
        self, batch: spectre_server.core.batches.CallistoBatch
    ) -> spectre_server.core.spectrograms.Spectrogram:
        """Create a CALLISTO-compatible spectrogram from a batch of IQ samples."""
        _LOGGER.info(f"Reading the I/Q samples")
        iq_data = batch.read_iq(self.__model.output_type)

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
            prepend_signal=self.__prepend_signal,
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

        linearised_digits = linear_calibration(
            dynamic_spectra, self.__model.callisto_scale_factor
        )

        _LOGGER.info("Creating the spectrogram")
        spectrogram = spectre_server.core.spectrograms.Spectrogram(
            linearised_digits,
            times,
            frequencies,
            spectre_server.core.spectrograms.SpectrumUnit.CALLISTO,
            batch.start_datetime,
        )

        spectrogram = spectre_server.core.spectrograms.time_average(
            spectrogram, max(self.__model.time_resolution, spectrogram.time_resolution)
        )
        spectrogram = spectre_server.core.spectrograms.frequency_average(
            spectrogram,
            max(self.__model.frequency_resolution, spectrogram.frequency_resolution),
        )

        _LOGGER.info("Spectrogram created successfully")

        # Cache the tail to prepend to the next batch. Copy so the full batch
        # isn't kept alive by the view.
        self.__prepend_signal = iq_data[
            -get_num_dangling_samples(self.__model.window_size) :
        ].copy()

        if not self.__model.keep_signal:
            _LOGGER.info(f"Deleting the I/Q samples")
            batch.delete_iq(self.__model.output_type)

        return spectrogram
