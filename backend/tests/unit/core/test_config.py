# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import tempfile
import os
import datetime
import typing

import pytest

import spectre_server.core.config


@pytest.fixture(autouse=True)
def patch_spectre_data_dir_path():
    """Patch the environment which defines the shared ancestral path for all data produced by Spectre."""
    pytest.MonkeyPatch().setenv(
        "SPECTRE_DATA_DIR_PATH", os.path.join("/tmp", ".spectre-data")
    )


@pytest.mark.parametrize(
    ["year", "month", "day", "expected_dir_path"],
    [
        (
            None,
            None,
            None,
            os.path.join("/tmp", ".spectre-data", "batches"),
        ),
        (
            2025,
            None,
            None,
            os.path.join("/tmp", ".spectre-data", "batches", "2025"),
        ),
        (
            2025,
            2,
            None,
            os.path.join("/tmp", ".spectre-data", "batches", "2025", "02"),
        ),
        (
            2025,
            2,
            13,
            os.path.join("/tmp", ".spectre-data", "batches", "2025", "02", "13"),
        ),
    ],
)
def test_get_batches_dir_path(
    year: int,
    month: int,
    day: int,
    expected_dir_path: str,
) -> None:
    """Check that the batches directory paths are created as expected."""
    result = spectre_server.core.config.paths.get_batches_dir_path(year, month, day)
    assert result == expected_dir_path


@pytest.mark.parametrize(
    ["scope", "expected_dir_path"],
    [
        (None, os.path.join("/tmp", ".spectre-data", "logs")),
        ("server", os.path.join("/tmp", ".spectre-data", "logs", "server")),
        ("worker", os.path.join("/tmp", ".spectre-data", "logs", "worker")),
    ],
)
def test_get_logs_dir_path(
    scope: typing.Optional[str],
    expected_dir_path: str,
) -> None:
    """Check that the logs directory paths are created as expected."""
    result = spectre_server.core.config.paths.get_logs_dir_path(scope)
    assert result == expected_dir_path


def test_get_configs_dir_path():
    """Check that the configs directory path is created as expected."""
    assert spectre_server.core.config.paths.get_configs_dir_path() == os.path.join(
        "/tmp", ".spectre-data", "configs"
    )


def test_set_spectre_data_dir_path():
    """Check that setting a new value of `SPECTRE_DATA_DIR_PATH` overrides the current value,
    and creates the appropriate directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        spectre_server.core.config.paths.set_spectre_data_dir_path(temp_dir)
        assert os.path.exists(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "batches"))
        assert os.path.exists(os.path.join(temp_dir, "logs"))
        assert os.path.exists(os.path.join(temp_dir, "configs"))


@pytest.mark.parametrize(
    ("format", "dt", "expected_parsed"),
    [
        (
            spectre_server.core.config.TimeFormat.DATE,
            "2025-01-11",
            datetime.datetime(2025, 1, 11),
        ),
        (
            spectre_server.core.config.TimeFormat.TIME,
            "23:59:59",
            datetime.datetime.strptime("23:59:59", "%H:%M:%S"),
        ),
        (
            spectre_server.core.config.TimeFormat.DATETIME,
            "2025-01-11T23:59:59.233Z",
            datetime.datetime(2025, 1, 11, 23, 59, 59, 233000),
        ),
    ],
)
def test_time_format_parsing(format: str, dt: str, expected_parsed: datetime.datetime):
    """Ensure that example datetimes parse correctly using the defined formats."""
    parsed = datetime.datetime.strptime(dt, format)
    assert parsed == expected_parsed


def test_utc_now_is_naive():
    """`utc_now` must return a timezone-naive datetime."""
    now = spectre_server.core.config.utc_now()
    assert now.tzinfo is None
