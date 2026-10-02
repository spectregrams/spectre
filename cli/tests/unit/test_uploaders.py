# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import shutil
import socket

import pytest

import spectre_cli.uploaders


class _DummyUploader(spectre_cli.uploaders.Uploader):
    """An uploader which "uploads" by copying into ``remote_dir``.

    Raises each of ``failures`` in turn before succeeding.
    """

    name = "Dummy"

    def __init__(
        self,
        remote_dir: pathlib.Path,
        failures: list[Exception],
        attempts: int = 3,
    ) -> None:
        super().__init__("host", 0, "username", "password", attempts, 0)
        self._remote_dir = remote_dir
        self._failures = list(failures)
        self.calls = 0

    def _upload_once(self, file_path: str) -> None:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        shutil.copy(file_path, self._remote_dir)


@pytest.fixture
def remote_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory standing in for the remote server."""
    path = tmp_path / "remote"
    path.mkdir()
    return path


@pytest.fixture
def file_path(tmp_path: pathlib.Path) -> str:
    """A real local file to upload."""
    path = tmp_path / "spectrogram.fit.gz"
    path.write_bytes(b"data")
    return str(path)


@pytest.fixture
def unused_port() -> int:
    """A local port which nothing is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestUploader:
    def test_upload_first_attempt(
        self, file_path: str, remote_dir: pathlib.Path
    ) -> None:
        """A successful upload delivers the file after exactly one attempt."""
        uploader = _DummyUploader(remote_dir, [])
        uploader.upload(file_path)
        assert uploader.calls == 1
        assert (remote_dir / "spectrogram.fit.gz").read_bytes() == b"data"

    def test_upload_retries_until_success(
        self, file_path: str, remote_dir: pathlib.Path
    ) -> None:
        """Transient failures are retried, so long as an attempt remains."""
        uploader = _DummyUploader(remote_dir, [OSError("boom"), EOFError()], attempts=3)
        uploader.upload(file_path)
        assert uploader.calls == 3
        assert (remote_dir / "spectrogram.fit.gz").read_bytes() == b"data"

    def test_upload_exhausts_attempts(
        self, file_path: str, remote_dir: pathlib.Path
    ) -> None:
        """When every attempt fails, raise an `UploadError` chained from the last failure."""
        last_failure = OSError("last")
        uploader = _DummyUploader(
            remote_dir, [OSError("first"), OSError("second"), last_failure], attempts=3
        )
        with pytest.raises(
            spectre_cli.uploaders.UploadError,
            match="could not upload to Dummy after 3 attempts",
        ) as exc_info:
            uploader.upload(file_path)
        assert uploader.calls == 3
        assert exc_info.value.__cause__ is last_failure
        assert list(remote_dir.iterdir()) == []

    def test_upload_does_not_retry_unexpected_errors(
        self, file_path: str, remote_dir: pathlib.Path
    ) -> None:
        """Errors which are not transient propagate immediately."""
        uploader = _DummyUploader(remote_dir, [ValueError("bug")])
        with pytest.raises(ValueError):
            uploader.upload(file_path)
        assert uploader.calls == 1
        assert list(remote_dir.iterdir()) == []

    def test_upload_missing_file(
        self, tmp_path: pathlib.Path, remote_dir: pathlib.Path
    ) -> None:
        """A missing file is reported by name, and never attempted."""
        uploader = _DummyUploader(remote_dir, [])
        with pytest.raises(FileNotFoundError, match="Not found: missing.fit.gz"):
            uploader.upload(str(tmp_path / "missing.fit.gz"))
        assert uploader.calls == 0
        assert list(remote_dir.iterdir()) == []


class TestServerUploaders:
    @pytest.mark.parametrize(
        "uploader_cls",
        [
            spectre_cli.uploaders.FhnwUploader,
            spectre_cli.uploaders.AstrodoncelUploader,
        ],
    )
    def test_unreachable_server(
        self,
        uploader_cls: type[spectre_cli.uploaders.Uploader],
        file_path: str,
        unused_port: int,
    ) -> None:
        """An unreachable server is reported as an `UploadError`."""
        uploader = uploader_cls("127.0.0.1", unused_port, "username", "password", 2, 0)
        with pytest.raises(
            spectre_cli.uploaders.UploadError,
            match=f"could not upload to {uploader.name} after 2 attempts",
        ):
            uploader.upload(file_path)
