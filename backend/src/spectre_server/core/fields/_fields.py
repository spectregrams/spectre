# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import typing

import pydantic


class Field:
    """Shared `pydantic` fields for all models."""

    max_noutput_items = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=1000,
            description="The maximum number of samples handled at each call of the work function in GNU Radio.",
        ),
    ]
    sample_rate = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The number of samples taken per second, in Hz.",
        ),
    ]
    batch_size = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="Signals are recorded in batches of this size, in seconds.",
        ),
    ]
    frequency = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="Frequency of the signal, in Hz.",
        ),
    ]
    amplitude = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Amplitude of the signal, arbitrary units.",
        ),
    ]
    window_size = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=2,
            description="The size of the window, in samples, when performing the Short Time FFT.",
        ),
    ]
    window_hop = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="How much the window is shifted, in samples, when performing the Short Time FFT.",
        ),
    ]
    window_type = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The type of window applied when performing the Short Time FFT.",
        ),
    ]
    center_frequency = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0.0,
            description="The center frequency, in Hz, determines the middle of the spectrogram frequency range.",
        ),
    ]
    frequency_resolution = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            ge=0,
            description="The minimum target frequency resolution of the spectrogram, in Hz. 0 for minimum possible.",
        ),
    ]
    time_resolution = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            ge=0,
            description="The minimum target time resolution of the spectrogram, in seconds. 0 for minimum possible.",
        ),
    ]
    time_range = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            ge=0,
            description="The minimum target time range of the spectrogram, in seconds. 0 for minimum possible.",
        ),
    ]
    origin = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword ORIGIN.",
        ),
    ]
    telescop = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword TELESCOP.",
        ),
    ]
    instrume = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword INSTRUME.",
        ),
    ]
    observer = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword OBSERVER.",
        ),
    ]
    object = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword OBJECT.",
        ),
    ]
    obsgeo_l = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword OBSGEO-L",
        ),
    ]
    obsgeo_h = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword OBSGEO-H.",
        ),
    ]
    obsgeo_b = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Corresponds to the FITS keyword OBSGEO-B.",
        ),
    ]
    keep_signal = typing.Annotated[
        bool,
        pydantic.Field(
            ...,
            validate_default=True,
            description="If True, keep the signal after creating the spectrogram. Otherwise, it is deleted from the file system.",
        ),
    ]
    frequency_hop = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The amount, in Hz, by which the center frequency is incremented for each step in the frequency sweep.",
        ),
    ]
    step_increment = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The length by which each step in the staircase is incremented.",
        ),
    ]
    min_samples_per_step = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The minimum number of samples in the shortest step of the staircase.",
        ),
    ]
    max_samples_per_step = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The maximum number of samples in the shortest step of the staircase.",
        ),
    ]
    bandwidth = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The range in Hz the signal will occupy in the frequency domain without attenutation.",
        ),
    ]
    if_gain = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The intermediate frequency gain, in dB. Negative value indicates attenuation.",
        ),
    ]
    rf_gain = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The radio frequency gain, in dB. Negative value indicates attenuation.",
        ),
    ]
    min_frequency = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The minimum center frequency, in Hz, for the frequency sweep.",
        ),
    ]
    max_frequency = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The maximum center frequency, in Hz, for the frequency sweep.",
        ),
    ]
    gain = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The gain value for the SDR, in dB",
        ),
    ]
    master_clock_rate = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The primary reference clock for the SDR, specified in Hz.",
        ),
    ]
    num_recv_frames = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The number of receive buffers to allocate",
            gt=0,
        ),
    ]
    buffers = typing.Annotated[
        int,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="Depth of the receive ring buffer.",
        ),
    ]
    wire_format = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Controls the form of the data over the bus/network.",
        ),
    ]
    dwell_time = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            gt=0,
            description="The amount of time spent at each center frequency in the sweep. This may vary slightly from what is specified due to the nature of GNU Radio runtime.",
        ),
    ]
    output_type = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The type of samples produced by the receiver.",
        ),
    ]
    antenna_port = typing.Annotated[
        str,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Specifies a particular antenna port on a receiver.",
        ),
    ]
    amp_on = typing.Annotated[
        bool,
        pydantic.Field(
            ...,
            validate_default=True,
            description="If true, amplify the signal.",
        ),
    ]
    bias_tee = typing.Annotated[
        bool,
        pydantic.Field(
            ...,
            validate_default=True,
            description="If true, enable the bias tee on the selected antenna port.",
        ),
    ]
    lna_gain = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The low-noise amplifier gain, in dB.",
        ),
    ]
    vga_gain = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="The variable-gain amplifier gain, in dB.",
        ),
    ]
    callisto_scale_factor = typing.Annotated[
        float,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Transform DFT amplitudes to linearised CALLISTO receiver digits through a linear calibration.",
        ),
    ]
    floor_start_times = typing.Annotated[
        bool,
        pydantic.Field(
            ...,
            validate_default=True,
            description="Floor spectrogram timestamps to the nearest multiple of `time_range` from midnight UTC. "
            "Ignored if `time_range` is not set.",
        ),
    ]
