# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import typing
import time
import dataclasses
import tempfile

import typer

from ._utils import (
    safe_request,
    safe_request_from_endpoint,
    get_config_file_name,
    spinner,
)
from ._secho_resources import (
    secho_new_resource,
    secho_stale_resource,
    secho_existing_resource,
)
from .get import download_callisto_resources
import spectre_cli.uploaders
from ..config import (
    FHNW_USERNAME,
    FHNW_PASSWORD,
    ASTRODONCEL_USERNAME,
    ASTRODONCEL_PASSWORD,
)

join_typer = typer.Typer(help="Join a network as a node.")


_UTC_TIME_FORMAT = "%H:%M:%S"
_UTC_DATETIME = "%Y-%m-%dT%H:%M:%S.%fZ"
_REQUIRED_MODE = "callisto"
_TIME_RANGE_MINUTES = 15
_UPLOAD_OFFSET_MINUTES = 1


def is_on_minute(t: datetime.time, minute: int) -> bool:
    return t.minute % minute == 0 and t.second == 0


def utc_combine(time: str, date: datetime.date) -> datetime.datetime:
    """Combine a UTC ``date`` and ``time`` string into a naive-UTC datetime."""
    as_time = datetime.datetime.strptime(time, _UTC_TIME_FORMAT).time()
    return datetime.datetime.combine(date, as_time)


def _utc_now() -> datetime.datetime:
    """Return the current UTC wall-clock time as a naive datetime."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def next_day(d: datetime.datetime) -> datetime.datetime:
    return d + datetime.timedelta(days=1)


def _validate_times(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    now: datetime.datetime,
    mod_minutes: int,
) -> None:
    """Check the start and end times make sense."""
    if start_time.tzinfo is not None or end_time.tzinfo is not None:
        typer.secho(
            f"Start and end times must be naive (interpreted as UTC).", fg="yellow"
        )
        raise typer.Exit(1)

    if start_time < now:
        typer.secho(
            f"Start time must be in the future.",
            fg="yellow",
        )
        raise typer.Exit(1)

    if end_time <= start_time:
        typer.secho(f"End time must be more than start time.")
        raise typer.Exit(1)

    if not is_on_minute(start_time.time(), mod_minutes) or not is_on_minute(
        end_time.time(), mod_minutes
    ):
        typer.secho(
            f"Error: Times must modulo {mod_minutes} minutes.",
            fg="yellow",
        )
        raise typer.Exit(1)


def _validate_config(
    tag: str,
    required_mode: str,
    expected_time_range: int,
) -> None:
    """Check the config is compatible with the e-Callisto network."""

    filename = get_config_file_name(None, tag)
    jsend_data = safe_request(f"spectre-data/configs/{filename}/raw", "GET")
    config = jsend_data["data"]

    mode = config["receiver_mode"]
    if mode != required_mode:
        typer.secho(
            f"Expected receiver mode '{required_mode}'."
            f"Got '{mode}' in config with tag '{tag}'."
        )
        raise typer.Exit(1)

    params = config["parameters"]

    def _get_param(param: str) -> str:
        if param not in params.keys():
            typer.secho(
                f"'{param}' is a required parameter. "
                f"Not found in config with tag '{tag}'."
            )
            raise typer.Exit(1)
        return params[param]

    time_range = float(_get_param("time_range"))
    if time_range != expected_time_range:
        typer.secho(
            f"e-Callisto requires spectrograms with time range '{expected_time_range}' seconds. "
            f"Got '{time_range}' in config with tag '{tag}'."
        )
        raise typer.Exit(1)

    if not _get_param("floor_start_times"):
        typer.secho(
            f"e-Callisto requires spectrogram time stamps to be floored. Please set 'floor_start_times'."
        )
        raise typer.Exit(1)

    # ``instrume`` is used in filenames for compatibility with e-Callisto.
    _ = _get_param("instrume")


def _wait_until_then(now: datetime.datetime, then: datetime.datetime) -> None:
    """Suspend program execution from ``now`` until ``then``."""
    with spinner(f"Waiting until {then.strftime(_UTC_DATETIME)}"):
        time.sleep((then - now).total_seconds())


def _expected_filename(tag: str, when: datetime.datetime) -> str:
    return f"{when.strftime(_UTC_DATETIME)}_{tag}.fit"


@dataclasses.dataclass(frozen=True)
class Upload:
    """At ``when`` export ``filename`` and upload it to the e-Callisto servers."""

    when: datetime.datetime
    filename: str


def make_upload_schedule(
    tag,
    start: datetime.datetime,
    end: datetime.datetime,
    time_range_minutes: int,
    upload_offset_minutes: int,
) -> list[Upload]:
    """Upload each spectrograms some offset after they were written to disk."""
    schedule: list[Upload] = []
    t = start + datetime.timedelta(minutes=time_range_minutes)
    while t <= end:
        schedule.append(
            Upload(
                t + datetime.timedelta(minutes=upload_offset_minutes),
                _expected_filename(
                    tag, t - datetime.timedelta(minutes=time_range_minutes)
                ),
            )
        )
        t += datetime.timedelta(minutes=time_range_minutes)
    return schedule


def _create_recording(tag: str, duration: float) -> str:
    jsend_dict = safe_request(
        "recordings",
        "POST",
        json={
            "tag": tag,
            "kind": "spectrogram",
            "duration": duration,
        },
    )
    return jsend_dict["data"]


def _stop_recording(
    endpoint: str,
) -> str:
    jsend_data = safe_request_from_endpoint(
        endpoint,
        "PATCH",
        json={"stop_requested": True},
    )
    return jsend_data["data"]


def _find_file(date: datetime.date, basename: str) -> typing.Optional[str]:
    params = {
        "year": date.year,
        "month": date.month,
        "day": date.day,
    }
    jsend_dict = safe_request(
        f"spectre-data/batches",
        "GET",
        params=params,
    )
    # Return the files endpoint, if it exists.
    for endpoint in jsend_dict["data"]:
        if basename in endpoint:
            return endpoint
    return None


def _require_credentials(
    server: str, username: typing.Optional[str], password: typing.Optional[str]
) -> tuple[str, str]:
    if username is None or password is None:
        typer.secho(f"{server} upload credentials are missing", fg="yellow")
        raise typer.Exit(1)
    return username, password


@join_typer.command(help="Join the e-Callisto network.")
def ecallisto(
    tag: str = typer.Option(
        ..., "--tag", "-t", help="The unique identifier of the config."
    ),
    start_time: str = typer.Option(
        ...,
        "--start-time",
        help="The start time of the observation (UTC), in the format `%H:%M:%S` (must be on a 15-minute boundary).",
    ),
    end_time: str = typer.Option(
        ...,
        "--end-time",
        help="The end time of the observation (UTC), in the format `%H:%M:%S` (must be on a 15-minute boundary).",
    ),
    end_next_day: bool = typer.Option(
        False,
        "--end-next-day",
        help="If provided, the end time is interpreted as on the next UTC day.",
    ),
    fhnw_host: str = typer.Option(
        "127.0.0.1",
        "--fhnw-host",
        help="FHNW FTP server host.",
    ),
    fhnw_port: int = typer.Option(
        2121,
        "--fhnw-port",
        help="FHNW FTP server port.",
    ),
    astrodoncel_host: str = typer.Option(
        "127.0.0.1",
        "--astrodoncel-host",
        help="Astrodoncel SFTP server host.",
    ),
    astrodoncel_port: int = typer.Option(
        2222,
        "--astrodoncel-port",
        help="Astrodoncel SFTP server port.",
    ),
) -> None:

    # Parse and validate the start and end dates.
    now = _utc_now()
    start_date = now.date()
    start = utc_combine(start_time, start_date)
    end_date = start_date if not end_next_day else next_day(now).date()
    end = utc_combine(end_time, end_date)
    _validate_times(start, end, now, _TIME_RANGE_MINUTES)

    # Make sure the config is compatible with e-Callisto.
    expected_time_range = _TIME_RANGE_MINUTES * 60
    _validate_config(tag, _REQUIRED_MODE, expected_time_range)

    # Fail fast if either server's credentials are missing.
    fhnw_username, fhnw_password = _require_credentials(
        "FHNW", FHNW_USERNAME, FHNW_PASSWORD
    )
    astrodoncel_username, astrodoncel_password = _require_credentials(
        "Astrodoncel", ASTRODONCEL_USERNAME, ASTRODONCEL_PASSWORD
    )
    uploaders: list[spectre_cli.uploaders.Uploader] = [
        spectre_cli.uploaders.FhnwUploader(
            fhnw_host, fhnw_port, fhnw_username, fhnw_password
        ),
        spectre_cli.uploaders.AstrodoncelUploader(
            astrodoncel_host,
            astrodoncel_port,
            astrodoncel_username,
            astrodoncel_password,
        ),
    ]

    # Repeat indefinitely, until a keyboard interrupt.
    while True:

        # Ahead of starting the recording, make the upload schedule.
        schedule = make_upload_schedule(
            tag, start, end, _TIME_RANGE_MINUTES, _UPLOAD_OFFSET_MINUTES
        )

        # Wait until the start time, then start the recording. Make sure it continues a little after the end time to give it time to write
        # the last spectrogram to disk.
        _wait_until_then(_utc_now(), start)
        buffer = _UPLOAD_OFFSET_MINUTES
        duration = (end - start + datetime.timedelta(minutes=buffer)).total_seconds()
        typer.secho("Starting...")
        recording_endpoint = _create_recording(tag, duration)
        secho_new_resource(recording_endpoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                for upload in schedule:
                    # Wait for the next upload.
                    now = _utc_now()
                    _wait_until_then(now, upload.when)

                    # Look for the file.
                    endpoint = _find_file(now.date(), upload.filename)
                    if endpoint is None:
                        # If we can't find it, abort.
                        secho_stale_resource(f"[not found] {upload.filename}")
                        break

                    # It's found - export and upload it.
                    file_paths = download_callisto_resources(
                        [endpoint], tmpdir, compress=True
                    )
                    for uploader in uploaders:
                        try:
                            uploader.upload(file_paths[0])
                        except spectre_cli.uploaders.UploadError as e:
                            secho_stale_resource(
                                f"[failed: {uploader.name}] {upload.filename}: {e}"
                            )
                        else:
                            secho_new_resource(
                                f"[uploaded: {uploader.name}] {upload.filename}"
                            )
            finally:
                # Make sure we don't unwittingly leave the recording running on error.
                typer.secho("Stopping...")
                _ = _stop_recording(recording_endpoint)
                secho_stale_resource(recording_endpoint)

            # Repeat the next day.
            start, end = next_day(start), next_day(end)
