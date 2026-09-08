# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime

import pytest

import numpy as np

import spectre_server.core.events
import spectre_server.core.fields


def is_close(a, b, atol=1e-5, rtol=0):
    """Uniform closeness check for arrays."""
    return np.allclose(a, b, atol=atol, rtol=rtol)


class TestSTFFT:
    @pytest.mark.parametrize(
        ("window_type", "window_size", "expected_result"),
        [
            (spectre_server.core.fields.WindowType.BOXCAR, 2, [1.0, 1.0]),
            (spectre_server.core.fields.WindowType.BOXCAR, 4, [1.0, 1.0, 1.0, 1.0]),
            (
                spectre_server.core.fields.WindowType.BOXCAR,
                8,
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            (spectre_server.core.fields.WindowType.HANN, 2, [0.0, 1.0]),
            (spectre_server.core.fields.WindowType.HANN, 4, [0.0, 0.5, 1.0, 0.5]),
            (
                spectre_server.core.fields.WindowType.HANN,
                8,
                [
                    0.0,
                    0.14644661,
                    0.5,
                    0.85355339,
                    1.0,
                    0.85355339,
                    0.5,
                    0.14644661,
                ],
            ),
            (spectre_server.core.fields.WindowType.BLACKMAN, 2, [0.0, 1.0]),
            (spectre_server.core.fields.WindowType.BLACKMAN, 4, [0.0, 0.34, 1.0, 0.34]),
            (
                spectre_server.core.fields.WindowType.BLACKMAN,
                8,
                [
                    0.0,
                    0.0664466094,
                    0.34,
                    0.773553391,
                    1.0,
                    0.773553391,
                    0.34,
                    0.0664466094,
                ],
            ),
        ],
    )
    def test_windows(
        self, window_type: str, window_size: int, expected_result: list[float]
    ) -> None:

        # Cast the window samples (hard-coded from Scipy v1.12.0) to 32-bit floats.
        expected = np.array(expected_result, dtype=np.float32)

        # Check that our own implementation is consistent with Scipy.
        actual = spectre_server.core.events.get_window(window_type, window_size)
        assert is_close(actual, expected)

    def test_compute_times(self) -> None:
        """Check that we assign the correct times to each spectrum."""
        num_spectrums = 4
        sample_rate = 2
        window_hop = 4
        expected_times = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
        assert is_close(
            spectre_server.core.events.get_times(
                num_spectrums, sample_rate, window_hop
            ),
            expected_times,
        )

    def test_compute_frequencies(self) -> None:
        """Check that we assign the correct frequencies to each spectral component."""
        window_size = 8
        sample_rate = 2
        expected_frequencies = [0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25]
        assert is_close(
            spectre_server.core.events.get_frequencies(window_size, sample_rate),
            expected_frequencies,
        )

    @pytest.mark.parametrize(
        ("signal_size", "window_size", "window_hop", "expected_num_spectrums"),
        [
            # Even signal size, even window size, even hop. Window size less than the hop.
            (8, 4, 2, 4),
            # Even signal size, even window size, even hop. Window size equal to the hop.
            (8, 4, 4, 2),
            # Even signal size, even window size, even hop. Window size greater than the hop.
            (8, 4, 6, 2),
            # Even signal size, even window size, odd hop. Window size less than the hop.
            (8, 4, 3, 3),
            # Even signal size, even window size, odd hop. Window size more than the hop.
            (8, 4, 5, 2),
            # Even signal size, odd window size, even hop. Window size less than the hop.
            (8, 3, 2, 4),
            # Even signal size, odd window size, even hop. Window size greater than the hop.
            (8, 3, 4, 2),
            # Even signal size, odd window size, odd hop. Window size less than the hop.
            (8, 5, 3, 2),
            # Even signal size, odd window size, odd hop. Window size equal to the hop.
            (8, 5, 5, 2),
            # Even signal size, odd window size, odd hop. Window size greater than the hop.
            (8, 3, 5, 2),
            # Odd signal size, even window size, even hop. Window size less than the hop.
            (9, 4, 2, 4),
            # Odd signal size, even window size, even hop. Window size equal to the hop.
            (9, 4, 4, 2),
            # Odd signal size, even window size, even hop. Window size greater than the hop.
            (9, 4, 6, 2),
            # Odd signal size, even window size, odd hop. Window size less than the hop.
            (9, 4, 3, 3),
            # Odd signal size, even window size, odd hop. Window size more than the hop.
            (9, 4, 5, 2),
            # Odd signal size, odd window size, even hop. Window size less than the hop.
            (9, 3, 2, 4),
            # Odd signal size, odd window size, even hop. Window size greater than the hop.
            (9, 3, 4, 2),
            # Odd signal size, odd window size, odd hop. Window size less than the hop.
            (9, 5, 3, 3),
            # Odd signal size, odd window size, odd hop. Window size equal to the hop.
            (9, 5, 5, 2),
            # Odd signal size, odd window size, odd hop. Window size greater than the hop.
            (9, 3, 5, 2),
            # Minimum window size.
            (8, 1, 4, 2),
            # Minimum hop size.
            (8, 3, 1, 7),
            # Minimum window size and minimum hop size.
            (8, 1, 1, 8),
            # Even signal size, maximum window size and hop.
            (8, 8, 4, 2),
            # Odd signal size, maximum window size and hop.
            (9, 9, 4, 2),
            # Window hop is greater than the signal size.
            (9, 9, 10, 1),
            # Window hop is much greater than the signal size.
            (9, 9, 999, 1),
            # Old test cases to make sure the new implementation is compatible with v1.1.1
            # At least, according to those test cases.
            (16, 8, 8, 2),
            (16, 8, 4, 4),
            (16, 7, 2, 7),
        ],
    )
    def test_num_spectrums(
        self,
        signal_size: int,
        window_size: int,
        window_hop: int,
        expected_num_spectrums: int,
    ) -> None:
        """Check that we compute the right number of spectrums for the stfft with given parameters."""
        assert expected_num_spectrums == spectre_server.core.events.get_num_spectrums(
            signal_size, window_size, window_hop
        )

    @pytest.mark.parametrize(
        ("signal_size", "window_size", "window_hop"),
        [
            # The window size cannot be less than one.
            (8, 0, 4),
            # The window hop cannot be less than one.
            (8, 4, 0),
            # The window must fit within the signal.
            (8, 9, 4),
        ],
    )
    def test_invalid_num_spectrums(
        self,
        signal_size: int,
        window_size: int,
        window_hop: int,
    ) -> None:
        """Check that passing bad arguments yields a ValueError in various cases."""
        with pytest.raises(ValueError):
            spectre_server.core.events.get_num_spectrums(
                signal_size, window_size, window_hop
            )

    @pytest.mark.parametrize(
        ("window_size", "window_hop", "expected"),
        [
            # Even window size, even hop. Hop greater than size.
            (4, 6, 1),
            # Even window size, even hop. Hop equals size.
            (4, 4, 1),
            # Even window size even hop. Hop half of size.
            (4, 2, 1),
            # Even window size, odd hop. Hop quater of size.
            (4, 1, 2),
            # Odd window size, odd hop. Hop equals size.
            (5, 5, 1),
            # Odd window size, even hop. Hop less than size.
            (5, 3, 1),
            # Odd window size, even hop. Hop less than size.
            (5, 2, 1),
            # Smallest possible window.
            (1, 1, 0),
            # Small window.
            (2, 1, 1),
            (2, 2, 1),
        ],
    )
    def test_num_dangling_windows(
        self,
        window_size: int,
        window_hop: int,
        expected: int,
    ) -> None:
        """Check that we compute the right number of dangling windows."""
        assert expected == spectre_server.core.events.get_num_dangling_windows(
            window_size, window_hop
        )

    @pytest.mark.parametrize(
        ("window_size", "window_hop"),
        [
            # The window size cannot be less than one.
            (0, 4),
            # The window hop cannot be less than one.
            (4, 0),
        ],
    )
    def test_invalid_num_dangling_windows(
        self,
        window_size: int,
        window_hop: int,
    ) -> None:
        """Check that passing bad arguments yields a ValueError in various cases."""
        with pytest.raises(ValueError):
            spectre_server.core.events.get_num_dangling_windows(window_size, window_hop)

    def test_stfft(self) -> None:
        """Check that the stfft of a simple cosine wave matches the analytically derived solution."""
        # Define the cosine wave.
        num_samples = 32
        sample_rate = 8
        frequency = 1
        phase = 0
        amplitude = 1

        # Define the window.
        window_type = spectre_server.core.fields.WindowType.BOXCAR
        window_size = 8
        window_hop = 8

        # Make the cosine signal, window and buffer.
        signal = spectre_server.core.events.get_cosine_signal(
            num_samples, sample_rate, frequency, amplitude, phase
        )
        window = spectre_server.core.events.get_window(window_type, window_size)
        buffer = spectre_server.core.events.get_buffer(window_size)

        # Plan, then compute the STFFT.
        fftw_obj = spectre_server.core.events.get_fftw_obj(buffer)
        dynamic_spectra = spectre_server.core.events.stfft(
            fftw_obj, buffer, signal, window, window_hop
        )

        # TODO: Replace this "point-in-time" check, with something more robust and human readable.
        # I'll go through the derivation again, and check against a runtime-computed analytical
        # solution.
        expected_dynamic_spectra = np.array(
            [
                [9.9999994e-01, 0.0, 0.0, 0.0],
                [2.0000000e00, 4.0, 4.0, 4.0],
                [1.7320508e00, 0.0, 0.0, 0.0],
                [7.3914812e-08, 0.0, 0.0, 0.0],
                [9.9999994e-01, 0.0, 0.0, 0.0],
                [7.3914812e-08, 0.0, 0.0, 0.0],
                [1.7320508e00, 0.0, 0.0, 0.0],
                [2.0000000e00, 4.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        )

        assert is_close(dynamic_spectra, expected_dynamic_spectra)
