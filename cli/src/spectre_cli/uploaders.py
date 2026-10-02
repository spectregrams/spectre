# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

"""Upload spectrograms to the servers of the e-Callisto network."""

import abc
import ftplib
import os
import os.path
import socket
import time
import typing

import paramiko

DEFAULT_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 5

_TIMEOUT_SECONDS = 10
_ASTRODONCEL_REMOTE_DIR = "/ftpfolder/"


class UploadError(Exception):
    """Raised when a file could not be uploaded after exhausting all attempts."""


class Uploader(abc.ABC):
    """Upload files to a remote server, retrying on transient failures."""

    name: typing.ClassVar[str]
    _RETRYABLE: typing.ClassVar[tuple[type[Exception], ...]] = (
        ftplib.Error,
        paramiko.SSHException,
        OSError,
        EOFError,
    )

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        attempts: int = DEFAULT_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        """
        :param host: The server host.
        :param port: The server port.
        :param username: The account username.
        :param password: The account password.
        :param attempts: The maximum number of upload attempts before giving up.
        :param retry_delay_seconds: The pause between consecutive attempts.
        """
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._attempts = attempts
        self._retry_delay_seconds = retry_delay_seconds

    @abc.abstractmethod
    def _upload_once(self, file_path: str) -> None:
        """Make a single attempt to upload the file at ``file_path``."""

    def upload(self, file_path: str) -> None:
        """Upload the file at ``file_path``, retrying on transient failures.

        :param file_path: The path to the local file.
        :raises FileNotFoundError: If ``file_path`` does not exist.
        :raises UploadError: If every attempt failed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Not found: {os.path.basename(file_path)}")

        for attempt in range(1, self._attempts + 1):
            try:
                self._upload_once(file_path)
                return
            except self._RETRYABLE as e:
                if attempt == self._attempts:
                    raise UploadError(
                        f"could not upload to {self.name} after {self._attempts} attempts"
                    ) from e
                time.sleep(self._retry_delay_seconds)


class FhnwUploader(Uploader):
    """Upload files to the FHNW FTP server."""

    name = "FHNW"

    def _upload_once(self, file_path: str) -> None:
        with ftplib.FTP(timeout=_TIMEOUT_SECONDS) as ftp:
            ftp.connect(self._host, self._port)
            ftp.login(self._username, self._password)
            ftp.set_pasv(True)

            with open(file_path, "rb") as f:
                basename = os.path.basename(file_path)
                tmpname = basename + ".tmp"
                ftp.storbinary("STOR " + tmpname, f)
                time.sleep(1)
                ftp.rename(tmpname, basename)


class AstrodoncelUploader(Uploader):
    """Upload files to the Astrodoncel SFTP server."""

    name = "Astrodoncel"

    def _upload_once(self, file_path: str) -> None:
        sock = socket.create_connection(
            (self._host, self._port), timeout=_TIMEOUT_SECONDS
        )
        with paramiko.Transport(sock) as transport:
            transport.connect(None, self._username, self._password)

            channel = transport.open_session(timeout=_TIMEOUT_SECONDS)
            channel.settimeout(_TIMEOUT_SECONDS)
            channel.invoke_subsystem("sftp")
            with paramiko.SFTPClient(channel) as sftp:
                basename = os.path.basename(file_path)
                remote_path = os.path.join(_ASTRODONCEL_REMOTE_DIR, basename)
                tmp_remote_path = remote_path + ".tmp"
                sftp.put(file_path, tmp_remote_path)
                time.sleep(1)
                sftp.rename(tmp_remote_path, remote_path)
