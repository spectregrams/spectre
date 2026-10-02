# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime

import pytest

import spectre_cli.commands.join


def _dt(hour: int, minute: int, second: int = 0) -> datetime.datetime:
    return datetime.datetime(2025, 1, 31, hour, minute, second)


class TestTimes:
    @pytest.mark.parametrize(
        ("t", "minute", "expected"),
        [
            (datetime.time(10, 0, 0), 15, True),
            (datetime.time(10, 15, 0), 15, True),
            (datetime.time(10, 45, 0), 15, True),
            (datetime.time(10, 10, 0), 15, False),
            (datetime.time(10, 15, 1), 15, False),
        ],
    )
    def test_is_on_minute(self, t: datetime.time, minute: int, expected: bool) -> None:
        """Times are on the minute boundary only if they have no seconds."""
        assert spectre_cli.commands.join.is_on_minute(t, minute) == expected

    def test_utc_combine(self) -> None:
        """Combine a date and a time string into a single naive datetime."""
        result = spectre_cli.commands.join.utc_combine(
            "10:15:30", datetime.date(2025, 1, 31)
        )
        assert result == _dt(10, 15, 30)
        assert result.tzinfo is None

    def test_utc_combine_invalid_format(self) -> None:
        """Times must be formatted as `%H:%M:%S`."""
        with pytest.raises(ValueError):
            spectre_cli.commands.join.utc_combine("10:15", datetime.date(2025, 1, 31))

    @pytest.mark.parametrize(
        ("d", "expected"),
        [
            (_dt(10, 0), datetime.datetime(2025, 2, 1, 10, 0)),
            (
                datetime.datetime(2025, 12, 31, 23, 45),
                datetime.datetime(2026, 1, 1, 23, 45),
            ),
        ],
    )
    def test_next_day(self, d: datetime.datetime, expected: datetime.datetime) -> None:
        """Advance by one day, rolling over month and year boundaries."""
        assert spectre_cli.commands.join.next_day(d) == expected


class TestMakeUploadSchedule:
    def test_schedule(self) -> None:
        """Each spectrogram is uploaded some offset after it is written to disk."""
        schedule = spectre_cli.commands.join.make_upload_schedule(
            "mytag", _dt(10, 0), _dt(10, 45), 15, 1
        )
        assert schedule == [
            spectre_cli.commands.join.Upload(
                _dt(10, 16), "2025-01-31T10:00:00.000000Z_mytag.fit"
            ),
            spectre_cli.commands.join.Upload(
                _dt(10, 31), "2025-01-31T10:15:00.000000Z_mytag.fit"
            ),
            spectre_cli.commands.join.Upload(
                _dt(10, 46), "2025-01-31T10:30:00.000000Z_mytag.fit"
            ),
        ]

    def test_schedule_shorter_than_time_range(self) -> None:
        """No spectrogram is completed if the observation is shorter than the time range."""
        schedule = spectre_cli.commands.join.make_upload_schedule(
            "mytag", _dt(10, 0), _dt(10, 10), 15, 1
        )
        assert schedule == []
