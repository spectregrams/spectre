# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeFormat:
    """Package-wide datetime formats.

    :ivar DATE: Format for dates (e.g., '2025-01-11').
    :ivar TIME: Format for times (e.g., '23:59:59').
    :ivar FRACTIONAL_TIME: Format for times with microsecond precision (e.g., '23:59:59.123456')
    :ivar DATETIME: Format for datetimes compliant with the ISO-8601 standard (e.g., '2025-01-11T23:59:59.123456Z')
    :ivar FITS: Format for ISO-8601 strings conforming to the FITS standard (e.g., '2025-01-11T23:59:59.123456')
    """

    DATE = "%Y-%m-%d"
    TIME = "%H:%M:%S"
    FRACTIONAL_TIME = "%H:%M:%S.%f"
    DATETIME = f"{DATE}T{FRACTIONAL_TIME}Z"
    # Time zone designators (like `Z`) are not allowed. See http://dx.doi.org/10.1051/0004-6361/201424653 section 3.1.
    FITS = f"%Y-%m-%dT%H:%M:%S.%f"


def utc_now() -> datetime.datetime:
    """Return the current UTC time as a naive datetime."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
