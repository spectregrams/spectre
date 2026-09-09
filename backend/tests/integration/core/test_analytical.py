# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import time
import typing
import pytest
import secrets

import spectre_server.core.receivers
import spectre_server.core.config
import spectre_server.core.batches
import spectre_server.services.recordings


@pytest.fixture
def signal_generator() -> spectre_server.core.receivers.SignalGenerator:
    """Get a signal generator, with mode not yet set."""
    return spectre_server.core.receivers.get_receiver("signal_generator")


ATOL = 1e-4
DURATION = 5
USE_DEFAULT_PARAMETERS: dict[str, typing.Any] = {}
COSINE_WAVE_MODE = "cosine_wave"
COSINE_WAVE_PARAMETERS = {
    "batch_size": 1,
    "amplitude": 3.0,
    "frequency": 16000.0,
    "window_hop": 256,
    "window_size": 256,
    "window_type": "boxcar",
    "sample_rate": 128000,
}

CONSTANT_STAIRCASE_MODE = "constant_staircase"
CONSTANT_STAIRCASE_PARAMETERS = {
    "batch_size": 1,
    "window_hop": 512,
    "window_size": 512,
    "window_type": "boxcar",
    "frequency_hop": 128000.0,
    "max_samples_per_step": 5000,
    "min_samples_per_step": 4000,
    "sample_rate": 128000,
    "step_increment": 200,
}


def _await_recording_finished(
    recording_id: str,
    paths: spectre_server.core.config.Paths,
    timeout_s: float = 30.0,
) -> dict[str, typing.Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        recording = spectre_server.services.recordings.get_recording(
            recording_id, db_path=paths.get_db_path()
        )
        if recording["state"] in {"completed", "failed"}:
            return recording
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for recording '{recording_id}' to finish")


def _validate_batches(
    config: spectre_server.core.receivers.Config,
    signal_generator: spectre_server.core.receivers.SignalGenerator,
    spectre_config_paths: spectre_server.core.config.Paths,
) -> None:
    signal_generator.mode = config.receiver_mode

    num_validated_batches = 0
    for batch in spectre_server.core.batches.Batches(
        config.tag,
        signal_generator.batch_cls,
        spectre_config_paths.get_batches_dir_path(),
    ):
        if not batch.spectrogram_file.exists:
            continue

        spectrogram = batch.read_spectrogram()
        result = signal_generator.validate_analytically(
            spectrogram,
            signal_generator.model_validate(config.parameters),
            ATOL,
        )
        assert result["frequencies_validated"]
        assert result["times_validated"]

        # The first batch should be the only one afflicted by window effects, and so (assuming the hop and size is such that
        # only one window is dangling in the stfft) we can accept at most one invalid spectrum in that case.
        max_invalid_spectrums = 1 if num_validated_batches == 0 else 0
        assert 0 <= result["num_invalid_spectrums"] <= max_invalid_spectrums

        num_validated_batches += 1

    # Check we've validated at least one spectrogram.
    assert num_validated_batches > 0


@pytest.mark.parametrize(
    ("modes", "parameters"),
    [
        ([COSINE_WAVE_MODE], [USE_DEFAULT_PARAMETERS]),
        ([COSINE_WAVE_MODE], [COSINE_WAVE_PARAMETERS]),
        ([CONSTANT_STAIRCASE_MODE], [USE_DEFAULT_PARAMETERS]),
        ([CONSTANT_STAIRCASE_MODE], [CONSTANT_STAIRCASE_PARAMETERS]),
        (
            [COSINE_WAVE_MODE, CONSTANT_STAIRCASE_MODE],
            [USE_DEFAULT_PARAMETERS, USE_DEFAULT_PARAMETERS],
        ),
    ],
)
def test_analytical(
    modes: list[str],
    parameters: list[dict[str, typing.Any]],
    spectre_config_paths: spectre_server.core.config.Paths,
    signal_generator: spectre_server.core.receivers.SignalGenerator,
) -> None:
    """Test end-to-end execution of the program using the signal generator, comparing
    the results to analytically derived solutions."""
    configs: list[spectre_server.core.receivers.Config] = []
    for mode, p in zip(modes, parameters):
        signal_generator.mode = mode
        rand_suffix = secrets.token_hex(2)
        tag = mode.replace("_", "-") + f"-{rand_suffix}"
        signal_generator.write_config(
            tag,
            p,
            configs_dir_path=spectre_config_paths.get_configs_dir_path(),
        )
        configs.append(
            signal_generator.read_config(
                tag, configs_dir_path=spectre_config_paths.get_configs_dir_path()
            )
        )

    recording_ids = [
        spectre_server.services.recordings.create_recording(
            tag=config.tag,
            kind="spectrogram",
            duration=DURATION,
            validate=True,
            paths=spectre_config_paths,
        )
        for config in configs
    ]

    for recording_id in recording_ids:
        recording = _await_recording_finished(recording_id, spectre_config_paths)
        assert recording["state"] == "completed"

    for config in configs:
        _validate_batches(config, signal_generator, spectre_config_paths)
