# SPDX-FileCopyrightText: © 2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import secrets
import time
import typing

import pytest

import spectre_server.core.batches
import spectre_server.core.config
import spectre_server.core.receivers
import spectre_server.services.recordings


@pytest.fixture
def signal_generator() -> spectre_server.core.receivers.SignalGenerator:
    """A signal generator with the ``cosine_wave`` mode selected."""
    receiver = spectre_server.core.receivers.get_receiver("signal_generator")
    receiver.mode = "cosine_wave"
    return receiver


def _await_recording_finished(
    recording_id: str,
    paths: spectre_server.core.config.Paths,
    timeout_s: float,
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


class TestBatchDriftRegression:
    def test_written_spectrograms_step_by_target_time_range(
        self,
        spectre_config_paths: spectre_server.core.config.Paths,
        signal_generator: spectre_server.core.receivers.SignalGenerator,
    ) -> None:
        """Guard against timestamp drift for spectrograms wrote to disk."""
        target_time_range = 3.0
        duration_s = 10.0
        parameters = {
            "batch_size": 1,
            "time_range": target_time_range,
            "amplitude": 3.0,
            "frequency": 16000.0,
            "window_hop": 256,
            "window_size": 256,
            "window_type": "boxcar",
            "sample_rate": 128000,
        }
        tag = f"cosine-wave-drift-{secrets.token_hex(2)}"
        signal_generator.write_config(
            tag,
            parameters,
            configs_dir_path=spectre_config_paths.get_configs_dir_path(),
        )

        recording_id = spectre_server.services.recordings.create_recording(
            tag=tag,
            kind="spectrogram",
            duration=duration_s,
            validate=True,
            paths=spectre_config_paths,
        )
        recording = _await_recording_finished(
            recording_id, spectre_config_paths, timeout_s=duration_s + 30.0
        )
        assert recording["state"] == "completed"

        start_datetimes = sorted(
            batch.start_datetime
            for batch in spectre_server.core.batches.Batches(
                tag,
                signal_generator.batch_cls,
                spectre_config_paths.get_batches_dir_path(),
            )
            if batch.spectrogram_file.exists
        )
        assert len(start_datetimes) == 3

        step = datetime.timedelta(seconds=target_time_range)
        assert start_datetimes[1] - start_datetimes[0] == step
        assert start_datetimes[2] - start_datetimes[1] == step
