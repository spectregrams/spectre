# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import time

import flask.testing

import spectre_server.core.config

TAG = "cw"


def test_plot_lifecycle(client: flask.testing.FlaskClient) -> None:
    """Create config, record spectrograms, plot them, fetch the plot via the API."""

    # Create a config.
    response = client.put(
        f"/spectre-data/configs/{TAG}.json",
        json={
            "receiver_name": "signal_generator",
            "receiver_mode": "cosine_wave",
            "validate": False,
        },
    )
    jsend = response.get_json()
    assert jsend["status"] == "success", jsend

    # Capture the current time window for the plot request.
    now = spectre_server.core.config.utc_now()
    obs_date = now.strftime("%Y-%m-%d")
    start_time = now.strftime("%H:%M:%S")

    # Start a short spectrogram recording.
    response = client.post(
        "/recordings/",
        json={
            "tag": TAG,
            "kind": "spectrogram",
            "duration": 5,
            "validate": False,
        },
    )
    jsend = response.get_json()
    assert jsend["status"] == "success", jsend
    recording_url = jsend["data"]

    # Wait for the recording to finish.
    time.sleep(10)

    response = client.get(recording_url)
    jsend = response.get_json()
    assert jsend["status"] == "success", jsend
    assert jsend["data"]["state"] == "completed", jsend

    end_time = (now + datetime.timedelta(seconds=10)).strftime("%H:%M:%S")

    # Create a plot.
    response = client.put(
        "/spectre-data/batches/plots",
        json={
            "tags": [TAG],
            "obs_date": obs_date,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    jsend = response.get_json()
    assert jsend["status"] == "success", jsend
    plot_url = jsend["data"]

    # Check a plot was created.
    response = client.get(plot_url)
    assert response.status_code == 200
    assert response.content_type == "image/png"
