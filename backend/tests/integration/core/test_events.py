# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import os

import numpy as np

import spectre_server.core.batches
import spectre_server.core.config
import spectre_server.core.events
import spectre_server.core.fields
import spectre_server.core.spectrograms

TAG = "fixed-center-frequency-test"
START = datetime.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)


def test_fixed_center_frequency(
    spectre_config_paths: spectre_server.core.config.Paths,
) -> None:
    """Check that ``FixedCenterFrequency.process`` produces the expected spectrogram, for constant IQ
    samples written directly to disk.
    """
    amplitude = 2.0
    window_size = 4
    window_hop = 4
    sample_rate = 4.0
    center_frequency = 100.0
    signal_size = 16

    model = spectre_server.core.events.FixedCenterFrequencyModel(
        window_size=window_size,
        window_hop=window_hop,
        window_type=spectre_server.core.fields.WindowType.BOXCAR,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        # Exercise averaging, too.
        time_resolution=2.0,
        frequency_resolution=2.0,
    )
    receiver = spectre_server.core.events.FixedCenterFrequency(
        TAG, model, spectre_server.core.batches.IQStreamBatch
    )

    batches_dir_path = spectre_config_paths.get_batches_dir_path(
        START.year, START.month, START.day
    )
    batch = spectre_server.core.batches.IQStreamBatch(
        batches_dir_path,
        datetime.datetime.strftime(
            START, spectre_server.core.config.TimeFormat.DATETIME
        ),
        TAG,
    )

    # Write some constant I/Q samples to disk.
    iq_data = np.full(signal_size, amplitude, dtype=np.complex64)
    os.makedirs(batch.fc32_file.parent_dir_path, exist_ok=True)
    iq_data.tofile(batch.fc32_file.file_path)

    # Compute the spectrogram.
    spectrogram = receiver.process(batch)

    # Compare with what's expected.
    expected_times = np.array([0.0, 2.0], dtype=np.float32)
    expected_frequencies = np.array([98.5, 100.5], dtype=np.float32)
    expected_dynamic_spectra = np.array(
        [
            [0.0, 0.0],
            [4.0, 4.0],
        ],
        dtype=np.float32,
    )

    # Check the data arrays.
    assert np.array_equal(spectrogram.times, expected_times)
    assert np.array_equal(spectrogram.frequencies, expected_frequencies)
    assert np.allclose(spectrogram.dynamic_spectra, expected_dynamic_spectra)

    # Check the resolutions are consistent with the model targets.
    assert np.isclose(model.time_resolution, spectrogram.time_resolution)
    assert np.isclose(model.frequency_resolution, spectrogram.frequency_resolution)

    # Check the spectrogram start time is consistent with the batch start time.
    assert spectrogram.start_datetime == np.datetime64(START)

    # Units should be DFT amplitude.
    assert (
        spectrogram.spectrum_unit
        == spectre_server.core.spectrograms.SpectrumUnit.AMPLITUDE
    )
