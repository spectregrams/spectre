# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import logging
import typing
import abc

import numpy as np
import pydantic
import watchdog.events

import spectre_server.core.spectrograms
import spectre_server.core.batches
import spectre_server.core.fields

_LOGGER = logging.getLogger(__name__)


def ready_to_flush(
    cached_time_range: float, batch_size: float, target_time_range: float
) -> bool:
    """Decide whether the cached spectrogram has covered the target duration.

    :param cached_time_range: The current ``time_range`` of the cached spectrogram.
    :param batch_size: The duration of one batch, in seconds.
    :param target_time_range: The target duration of the written spectrogram, in seconds.
    :return: True when adding another batch would overshoot the target.
    """
    return cached_time_range + batch_size > target_time_range


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        validate_assignment=True,
    )
    time_range: spectre_server.core.fields.Field.time_range = 0
    batch_size: spectre_server.core.fields.Field.batch_size = 3
    origin: spectre_server.core.fields.Field.origin = "NOTSET"
    telescop: spectre_server.core.fields.Field.telescop = "NOTSET"
    instrume: spectre_server.core.fields.Field.instrume = "NOTSET"
    observer: spectre_server.core.fields.Field.observer = "NOTSET"
    object: spectre_server.core.fields.Field.object = "NOTSET"
    obsgeo_b: spectre_server.core.fields.Field.obsgeo_b = 0.0
    obsgeo_l: spectre_server.core.fields.Field.obsgeo_l = 0.0
    obsgeo_h: spectre_server.core.fields.Field.obsgeo_h = 0.0
    floor_start_times: spectre_server.core.fields.Field.floor_start_times = False


B = typing.TypeVar("B", bound=spectre_server.core.batches.Base)
M = typing.TypeVar("M", bound=BaseModel)


class Base(abc.ABC, typing.Generic[M, B], watchdog.events.FileSystemEventHandler):
    def __init__(
        self,
        tag: str,
        model: M,
        batch_cls: typing.Type[B],
        queued_file: typing.Optional[str] = None,
        cached_spectrogram: typing.Optional[
            spectre_server.core.spectrograms.Spectrogram
        ] = None,
    ) -> None:
        """An abstract interface enabling event-driven file processing.

        :param tag: The data tag.
        :param model: Defines configurable parameters.
        :param batch_cls: The batch used to read data files.
        :param queued_file: Optionally override the queued file, defaults to None
        :param cached_spectrogram: Optionally override the cached spectrogram, defaults to None
        """
        self._tag = tag
        self.__batch_cls = batch_cls
        self.__model = model
        self.__queued_file = queued_file
        self.__cached_spectrogram = cached_spectrogram

    @abc.abstractmethod
    def process(self, batch: B) -> spectre_server.core.spectrograms.Spectrogram:
        """Transform data from the input batch into a spectrogram.

        :param batch: The batch providing an interface to read signal data from the filesystem.
        :return: The signal data transformed into a spectrogram.
        """

    @property
    @abc.abstractmethod
    def _watch_extension(self) -> str:
        """Newly created files with this extension trigger the batch to be processed."""

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        """Process a newly created batch file, only once the next batch is created.

        Since we assume that the batches are non-overlapping in time, this guarantees
        we avoid post processing a file while it is being written to. Files are processed
        sequentially, in the order they are created.

        :param event: The file system event containing the file details.
        """
        # The `src_path`` attribute holds the absolute path of the freshly closed file
        absolute_file_path = event.src_path

        # Only process a file if:
        #
        # - It's extension matches the `watch_extension` as defined in the config.
        # - It's tag matches the current sessions tag.
        #
        # This is important for two reasons.
        #
        # In the case of one session, the capture worker may write to two batch files simultaneously
        # (e.g., raw data file + seperate metadata file). We want to process them together - but this method will get called
        # seperately for both file creation events. So, we filter by extension to account for this.
        #
        # Additionally in the case of multiple sessions, the capture workers will create batch files in the same directory concurrently.
        # This method is triggered for all file creation events, so we ensure the batch file tag matches the session tag and early return
        # otherwise. This way, each post processor worker picks up the right files to process.
        if not absolute_file_path.endswith(f"_{self._tag}.{self._watch_extension}"):
            return

        _LOGGER.info(f"Noticed {absolute_file_path}")
        # If there exists a queued file, try and process it
        if self.__queued_file is not None:
            try:
                _LOGGER.info(f"Processing {self.__queued_file}")
                batches_dir_path, start_time, tag, _ = (
                    spectre_server.core.batches.parse_batch_file_path(
                        self.__queued_file
                    )
                )
                spectrogram = self.process(
                    self.__batch_cls(batches_dir_path, start_time, tag)
                )
                self.__cache_spectrogram(spectrogram)
            except Exception:
                _LOGGER.error(
                    f"An error has occured while processing {self.__queued_file}",
                    exc_info=True,
                )
                # Flush any internally stored spectrogram on error to avoid lost data
                self.__flush_cache()
                # re-raise the exception to the main thread
                raise

        # Queue the current file for processing next
        _LOGGER.info(f"Queueing {absolute_file_path} for post processing")
        self.__queued_file = absolute_file_path

    def __cache_spectrogram(
        self, spectrogram: spectre_server.core.spectrograms.Spectrogram
    ) -> None:
        if self.__cached_spectrogram is None:
            self.__cached_spectrogram = spectrogram
        else:
            self.__cached_spectrogram = (
                spectre_server.core.spectrograms.join_spectrograms(
                    [self.__cached_spectrogram, spectrogram]
                )
            )

        if ready_to_flush(
            self.__cached_spectrogram.time_range,
            self.__model.batch_size,
            self.__model.time_range,
        ):
            self.__flush_cache()

    def __flush_cache(self) -> None:
        if self.__cached_spectrogram:
            _LOGGER.info(
                f"Writing spectrogram to disk with start time "
                f"'{self.__cached_spectrogram.format_start_time()}'"
            )

            # We can only floor start times when the time range is set.
            floor_to_time_range: typing.Optional[float] = None
            if self.__model.floor_start_times and self.__model.time_range > 0:
                floor_to_time_range = self.__model.time_range

            # Use the start time of the spectrogram to make a new batch to write to.
            batch = spectre_server.core.batches.from_spectrogram(
                self.__batch_cls,
                self._tag,
                self.__cached_spectrogram,
                floor_to_time_range=floor_to_time_range,
            )

            batch.write_spectrogram(
                self.__cached_spectrogram,
                self.__model.origin,
                self.__model.instrume,
                self.__model.observer,
                self.__model.object,
                self.__model.telescop,
                self.__model.obsgeo_b,
                self.__model.obsgeo_l,
                self.__model.obsgeo_h,
            )

            # Reset the cache.
            self.__cached_spectrogram = None
