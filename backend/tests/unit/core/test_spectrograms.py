# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
import datetime

import spectre_server.core.spectrograms


@pytest.fixture
def spectrogram() -> spectre_server.core.spectrograms.Spectrogram:
    """Create the following spectrogram:

    1MHz  | 0    1    2    3   4   5   |
    2MHz  | 6    7    8    9   10  11  |
    3MHz  | 12   13   14   15  16  17  |
    4MHz  | 18   19   20   21  22  23  |
            0.0  0.2  0.4  0.6 0.8 1.0 [s]
    """
    dynamic_spectra = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
        ],
        dtype=np.float32,
    )
    times = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.0])
    frequencies = np.array([1e6, 2e6, 3e6, 4e6])
    return spectre_server.core.spectrograms.Spectrogram(
        dynamic_spectra,
        times,
        frequencies,
        spectre_server.core.spectrograms.SpectrumUnit.AMPLITUDE,
    )


class TestTimeAverage:
    def test_resolution_too_small(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check an error is raised when the desired time resolution is less than the current."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.time_average(spectrogram, 0.1)

    def test_resolution_too_big(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check an error is raised when the desired time resolution is more than the time spanned by the spectrogram."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.time_average(spectrogram, 1.0)

    def test_no_change_at_current_resolution(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that we get back the spectrogram unchanged when the time resolution is equal to the current."""
        averaged_s = spectre_server.core.spectrograms.time_average(spectrogram, 0.2)
        assert spectrogram.time_resolution == averaged_s.time_resolution
        assert np.array_equal(averaged_s.times, spectrogram.times)
        assert np.array_equal(averaged_s.frequencies, spectrogram.frequencies)
        assert np.array_equal(averaged_s.dynamic_spectra, spectrogram.dynamic_spectra)

    @pytest.mark.parametrize(
        "resolution, expected_resolution, expected_dynamic_spectra, expected_times",
        [
            # Impossible resolution, moving average divides 6 spectra into 3 full windows.
            pytest.param(
                0.45,
                0.4,
                [
                    [0.5, 2.5, 4.5],
                    [6.5, 8.5, 10.5],
                    [12.5, 14.5, 16.5],
                    [18.5, 20.5, 22.5],
                ],
                [0, 0.4, 0.8],
            ),
            # Impossible resolution, moving average divides 6 spectra into 1 full window, 1 partial.
            pytest.param(
                0.85,
                0.8,
                [[1.5, 4.5], [7.5, 10.5], [13.5, 16.5], [19.5, 22.5]],
                [0, 0.8],
            ),
            # Exact resolution, moving average divides 6 spectra into 1 full window, 1 partial.
            pytest.param(
                0.8,
                0.8,
                [[1.5, 4.5], [7.5, 10.5], [13.5, 16.5], [19.5, 22.5]],
                [0, 0.8],
            ),
        ],
    )
    def test_averaging(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
        resolution: float,
        expected_resolution: float,
        expected_dynamic_spectra: list[list[float]],
        expected_times: list[float],
    ) -> None:
        """Check that time averaging yields the correct resolution, dynamic spectra, times, and frequencies."""
        averaged_s = spectre_server.core.spectrograms.time_average(
            spectrogram, resolution
        )
        assert averaged_s.time_resolution == expected_resolution
        assert np.allclose(
            averaged_s.dynamic_spectra,
            np.array(expected_dynamic_spectra, dtype=np.float32),
        )
        assert np.allclose(averaged_s.times, np.array(expected_times, dtype=np.float32))
        assert np.allclose(averaged_s.frequencies, spectrogram.frequencies)


class TestFrequencyAverage:
    def test_resolution_too_small(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check an error is raised when the desired frequency resolution is less than the current."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.frequency_average(spectrogram, 0.5e6)

    def test_resolution_too_big(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check an error is raised when the desired frequency resolution is more than the frequency spanned by the spectrogram."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.frequency_average(spectrogram, 3e6)

    def test_no_change_at_current_resolution(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that we get back the spectrogram unchanged when the frequency resolution is equal to the current."""
        averaged_s = spectre_server.core.spectrograms.frequency_average(
            spectrogram, 1e6
        )
        assert spectrogram.frequency_resolution == averaged_s.frequency_resolution
        assert np.array_equal(averaged_s.frequencies, spectrogram.frequencies)
        assert np.array_equal(averaged_s.times, spectrogram.times)
        assert np.array_equal(averaged_s.dynamic_spectra, spectrogram.dynamic_spectra)

    @pytest.mark.parametrize(
        "resolution, expected_resolution, expected_dynamic_spectra, expected_frequencies",
        [
            # Impossible resolution, moving average divides 4 frequency bins into 2 full windows.
            pytest.param(
                2.5e6,
                2e6,
                [
                    [3, 4, 5, 6, 7, 8],
                    [15, 16, 17, 18, 19, 20],
                ],
                [1.5e6, 3.5e6],
            ),
            # Exact resolution, moving average divides 4 frequency bins into 2 full windows.
            pytest.param(
                2e6,
                2e6,
                [
                    [3, 4, 5, 6, 7, 8],
                    [15, 16, 17, 18, 19, 20],
                ],
                [1.5e6, 3.5e6],
            ),
        ],
    )
    def test_averaging(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
        resolution: float,
        expected_resolution: float,
        expected_dynamic_spectra: list[list[float]],
        expected_frequencies: list[float],
    ) -> None:
        """Check that frequency averaging yields the correct resolution, dynamic spectra, frequencies, and times."""
        averaged_s = spectre_server.core.spectrograms.frequency_average(
            spectrogram, resolution
        )
        assert averaged_s.frequency_resolution == expected_resolution
        assert np.allclose(
            averaged_s.dynamic_spectra,
            np.array(expected_dynamic_spectra, dtype=np.float32),
        )
        assert np.allclose(
            averaged_s.frequencies,
            np.array(expected_frequencies, dtype=np.float32),
        )
        assert np.allclose(averaged_s.times, spectrogram.times)


class TestMovingAverage:
    @pytest.mark.parametrize(
        ("current", "target", "expected"),
        [
            (0.25, 0.5, 2),
            (0.25, 0.25, 1),
            (0.25, 0.1, 1),
            (0.25, 0.33, 1),
            (0.25, 0.75, 3),
        ],
    )
    def test_get_moving_average_window_size(
        self, current: float, target: float, expected: int
    ) -> None:
        """Make sure we properly compute the size of the window in the moving average to achieve some target resolution."""
        assert (
            spectre_server.core.spectrograms.get_moving_average_window_size(
                target, current
            )
            == expected
        )

    def test_negative_resolution_raises(self) -> None:
        """Check that trying to determine the moving average window size with negative resolutions raises"""
        with pytest.raises(ValueError, match="negative resolutions"):
            spectre_server.core.spectrograms.get_moving_average_window_size(-1, -1)


class TestIgnoreNLeading:
    def test_zero_returns_unchanged(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that n=0 yields an identical dynamic spectra, with no NaNs introduced."""
        transformed_s = spectre_server.core.spectrograms.ignore_leading_spectrums(
            spectrogram, 0
        )
        assert np.array_equal(
            transformed_s.dynamic_spectra, spectrogram.dynamic_spectra
        )
        assert not np.any(np.isnan(transformed_s.dynamic_spectra))

        # Other metadata is unchanged.
        assert np.array_equal(transformed_s.times, spectrogram.times)
        assert np.array_equal(transformed_s.frequencies, spectrogram.frequencies)
        assert transformed_s.spectrum_unit == spectrogram.spectrum_unit
        assert transformed_s.start_datetime_is_set == spectrogram.start_datetime_is_set

    def test_nan_leading(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that n=2 NaNs the first two columns, leaving the rest (and metadata) unchanged."""
        transformed_s = spectre_server.core.spectrograms.ignore_leading_spectrums(
            spectrogram, 2
        )
        assert np.all(np.isnan(transformed_s.dynamic_spectra[:, :2]))
        assert np.array_equal(
            transformed_s.dynamic_spectra[:, 2:], spectrogram.dynamic_spectra[:, 2:]
        )

        # Other metadata is unchanged.
        assert np.array_equal(transformed_s.times, spectrogram.times)
        assert np.array_equal(transformed_s.frequencies, spectrogram.frequencies)
        assert transformed_s.spectrum_unit == spectrogram.spectrum_unit
        assert transformed_s.start_datetime_is_set == spectrogram.start_datetime_is_set

    def test_negative_n_raises(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that a negative n raises a ValueError."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.ignore_leading_spectrums(spectrogram, -1)

    @pytest.mark.parametrize("n", [7, 8])
    def test_n_exceeds_num_spectrums_raises(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
        n: int,
    ) -> None:
        """Check that n equal to or greater than the spectrum count raises a ValueError."""
        with pytest.raises(ValueError):
            spectre_server.core.spectrograms.ignore_leading_spectrums(spectrogram, n)

    def test_composes_with_time_average(
        self,
        spectrogram: spectre_server.core.spectrograms.Spectrogram,
    ) -> None:
        """Check that a NaN'd leading spectrum is excluded from a subsequent time average."""
        transformed_s = spectre_server.core.spectrograms.ignore_leading_spectrums(
            spectrogram, 1
        )
        averaged_s = spectre_server.core.spectrograms.time_average(transformed_s, 0.4)
        # The first spectrum is excluded from averaging, so the first spectrum in the averaged
        # spectrogram assumes the value of the second spectrum in the original.
        assert np.allclose(
            averaged_s.dynamic_spectra[:, 0], spectrogram.dynamic_spectra[:, 1]
        )
        assert not np.any(np.isnan(averaged_s.dynamic_spectra))


class TestSpectrogram:
    def test_start_datetime_setter(
        self, spectrogram: spectre_server.core.spectrograms.Spectrogram
    ) -> None:
        """Check that we can set the start datetime of a spectrogram."""
        dt = datetime.datetime(year=2000, month=1, day=1)
        assert not spectrogram.start_datetime_is_set
        spectrogram.start_datetime = np.datetime64(dt)
        assert spectrogram.start_datetime_is_set
        assert spectrogram.start_datetime == dt
