# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import logging
import time
import typing

import spectre_server.core.config

from ._process_types import ProcessType


def configure_root_logger(
    file_path: str,
    level: int = logging.INFO,
) -> str:
    """Configure the root logger to write log events to the given file.

    The parent directory of `file_path` is created if it does not exist.

    :param file_path: Absolute path of the log file to write to.
    :param level: The logging level, defaults to logging.INFO.
    :return: The absolute file path of the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)8s] --- %(message)s (%(name)s:%(lineno)s)",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return os.path.abspath(file_path)


def get_root_logger_state() -> tuple[bool, int]:
    """Get the state of the root logger.

    :return: A tuple containing:
        - A boolean indicating whether the root logger has any handlers.
        - The logging level of the root logger.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return True, root_logger.level
    return False, logging.NOTSET


def get_server_log_file_path(
    start_time: str,
    pid: int,
    logs_dir_path: typing.Optional[str] = None,
) -> str:
    """Get the absolute path of a server log file.

    :param start_time: When the process started.
    :param pid: The OS process id.
    :param logs_dir_path: If specified, place the log directly under this directory. Otherwise defaults to the server-scoped log subdirectory.
    :return: The absolute path of the log file.
    """
    directory = logs_dir_path or spectre_server.core.config.paths.get_logs_dir_path(
        ProcessType.SERVER.value
    )
    return os.path.join(directory, f"{start_time}_{pid}.log")


def get_worker_log_file_path(
    recording_id: str,
    worker_name: str,
    logs_dir_path: typing.Optional[str] = None,
) -> str:
    """Get the absolute path of a worker log file.

    :param recording_id: Parent recording identifier.
    :param worker_name: The worker's name.
    :param logs_dir_path: If specified, place the log directly under this directory. Otherwise defaults to the worker-scoped log subdirectory.
    :return: The absolute path of the log file.
    """
    directory = logs_dir_path or spectre_server.core.config.paths.get_logs_dir_path(
        ProcessType.WORKER.value
    )
    return os.path.join(directory, f"{recording_id}_{worker_name}.log")
