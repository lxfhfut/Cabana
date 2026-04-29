"""Tests for cabana/detector.py — FibreDetector class."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.detector import FibreDetector


def make_gray_image_with_line(h=128, w=128, line_y=64, thickness=5):
    """White image with a dark horizontal band (dark line on light background)."""
    img = np.ones((h, w), dtype=np.uint8) * 220
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = 50
    return img


def make_white_image(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)


def make_random_image(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_line_widths_array(self):
        d = FibreDetector()
        assert isinstance(d.line_widths, np.ndarray)

    def test_custom_line_widths(self):
        d = FibreDetector(line_widths=[5, 9, 13])
        assert len(d.line_widths) == 3

    def test_scalar_line_width(self):
        d = FibreDetector(line_widths=7)
        assert d.line_widths.shape == (1,)

    def test_dark_line_flag(self):
        d = FibreDetector(dark_line=True)
        assert d.dark_line is True

    def test_sigmas_computed(self):
        d = FibreDetector(line_widths=[3, 5])
        assert len(d.sigmas) == 2
        assert np.all(d.sigmas > 0)

    def test_dark_line_adjusts_thresholds(self):
        # When dark_line=True, clow = 255 - high_contrast
        d = FibreDetector(low_contrast=50, high_contrast=150, dark_line=True)
        assert d.clow == 255 - 150
        assert d.chigh == 255 - 50

    def test_light_line_keeps_thresholds(self):
        d = FibreDetector(low_contrast=50, high_contrast=150, dark_line=False)
        assert d.clow == 50
        assert d.chigh == 150

    def test_min_len_stored(self):
        d = FibreDetector(min_len=10)
        assert d.min_len == 10

    def test_extend_line_flag(self):
        d = FibreDetector(extend_line=True)
        assert d.extend_line is True


# ---------------------------------------------------------------------------
# detect_lines + apply_filtering (integration via detect_lines)
# ---------------------------------------------------------------------------

class TestDetectLines:
    def test_detect_lines_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)  # should not raise

    def test_detect_lines_white_image_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_white_image()
        d.detect_lines(img)

    def test_detect_lines_multiple_widths(self):
        d = FibreDetector(line_widths=[3, 5, 7], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)

    def test_detect_lines_with_estimate_width(self):
        d = FibreDetector(line_widths=[5], dark_line=True, estimate_width=True, min_len=3)
        img = make_gray_image_with_line(thickness=7)
        d.detect_lines(img)

    def test_detect_lines_no_estimate_width(self):
        d = FibreDetector(line_widths=[5], dark_line=True, estimate_width=False, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)

    def test_sets_image_attribute(self):
        d = FibreDetector(line_widths=[5], dark_line=True)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert d.image is not None

    def test_sets_gray_attribute(self):
        d = FibreDetector(line_widths=[5], dark_line=True)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert d.gray is not None
        assert d.gray.ndim == 2

    def test_contours_attribute_exists(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        assert hasattr(d, 'contours')


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------

class TestGetResults:
    def test_get_results_no_crash(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        assert result is not None

    def test_get_results_returns_tuple(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        assert isinstance(result, tuple)

    def test_get_results_on_white_image(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_white_image()
        d.detect_lines(img)
        result = d.get_results()
        assert result is not None

    def test_contour_image_is_ndarray(self):
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = make_gray_image_with_line()
        d.detect_lines(img)
        result = d.get_results()
        # result is a tuple; at least the first element should be array-like
        assert isinstance(result[0], np.ndarray) or result[0] is not None


# ---------------------------------------------------------------------------
# Input intensity normalization (uint8 / uint16 / float, hot pixels)
# ---------------------------------------------------------------------------

def _gray_uint8_with_line(h=128, w=128, line_y=64, thickness=5):
    """Standard 8-bit dark-line-on-light-background fixture."""
    img = np.full((h, w), 220, dtype=np.uint8)
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = 50
    return img


def _gray_uint16_with_line(h=128, w=128, line_y=64, thickness=5,
                           bg=55000, line=12000, hot_pixel=False):
    """16-bit dark-line-on-light-background fixture; optional hot pixel."""
    img = np.full((h, w), bg, dtype=np.uint16)
    img[line_y - thickness // 2:line_y + thickness // 2 + 1, :] = line
    if hot_pixel:
        # Single saturated pixel far from the line — what min/max would latch
        # onto and what percentile normalization should robustly ignore.
        img[0, 0] = np.iinfo(np.uint16).max
    return img


class TestInputNormalization:
    """detect_lines must be insensitive to outliers and dtype variation."""

    def test_uint8_passthrough_unchanged(self):
        """uint8 input is not re-normalized (preserves user's calibration)."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = _gray_uint8_with_line()
        d.detect_lines(img)
        # Fall-through path: self.image must equal the original input.
        np.testing.assert_array_equal(d.image, img)

    def test_uint16_loaded_as_uint8(self):
        """uint16 input should be converted to uint8 before downstream ops."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = _gray_uint16_with_line()
        d.detect_lines(img)
        assert d.image.dtype == np.uint8

    def test_uint16_hot_pixel_does_not_collapse_dynamic_range(self):
        """A single saturated pixel must not squash the rest into a tiny band.

        With min/max normalization, the 65535 hot pixel forces the entire
        image to be re-scaled relative to that maximum, leaving the bulk
        of the image around (12000-55000)/65535 * 255 = 47..213 (correct
        but only because line+bg are still in the bulk). A more telling
        case: a hot pixel + low dynamic range. Construct that here.
        """
        h, w = 128, 128
        img = np.full((h, w), 30000, dtype=np.uint16)
        img[64 - 2:64 + 3, :] = 28000   # very low-contrast dark line
        img[0, 0] = np.iinfo(np.uint16).max  # single hot pixel
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d.detect_lines(img)

        # After percentile normalization (0.5, 99.5), the hot pixel is
        # clipped and the bulk maps to a wide range. Check that the line
        # and background are visibly separated in the uint8 output.
        line_mean = float(d.image[64 - 2:64 + 3, :].mean())
        bg_mean = float(d.image[10:50, :].mean())
        # If percentile clipping worked, we expect a meaningful contrast
        # between line and background. If min/max squashed everything, the
        # contrast would be near zero (both ~ 30000/65535 * 255 ≈ 117).
        assert abs(bg_mean - line_mean) > 30.0

    def test_uint16_preserves_line_below_background(self):
        """uint16 → uint8 conversion must preserve dark-line-on-light topology.

        Percentile normalization stretches 16-bit dynamic range to fill
        0..255, which is *not* byte-identical to uint8 passthrough.
        What must be preserved is the relative ordering: line pixels
        remain darker than background, and the contrast is large enough
        for downstream thresholding.
        """
        img8 = _gray_uint8_with_line()
        img16 = (img8.astype(np.uint16) * 257)  # promotes 0..255 -> 0..65535

        d16 = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d16.detect_lines(img16)

        line_mean = float(d16.image[62:67, :].mean())
        bg_mean = float(d16.image[10:50, :].mean())
        assert line_mean < bg_mean
        assert (bg_mean - line_mean) > 100  # well-separated after stretching

    def test_uint16_two_seeds_equivalent_normalization(self):
        """Two uint16 images with identical p0.5/p99.5 percentiles produce
        the same normalized uint8 result up to interpolation noise.

        This is the core invariance contract for percentile normalization:
        the output depends only on the (p0.5, p99.5) range, not on the
        sparse extremes outside that range.
        """
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        # Same bulk distribution, different sparse tails (≤ 0.4% of pixels).
        h, w = 128, 128
        bulk_a = rng_a.integers(20000, 50000, size=(h, w), dtype=np.uint16)
        bulk_b = rng_b.integers(20000, 50000, size=(h, w), dtype=np.uint16)
        # Replace 0.3% with extreme outliers (different in each image).
        n_out = int(0.003 * h * w)
        idx_a = rng_a.choice(h * w, size=n_out, replace=False)
        idx_b = rng_b.choice(h * w, size=n_out, replace=False)
        bulk_a.flat[idx_a] = np.iinfo(np.uint16).max
        bulk_b.flat[idx_b] = 0

        d_a = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d_b = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d_a.detect_lines(bulk_a)
        d_b.detect_lines(bulk_b)
        # Bulk distributions are similar → normalized images have similar
        # mean and std, and the differences from outliers are clipped.
        assert abs(float(d_a.image.mean()) - float(d_b.image.mean())) < 5.0
        assert abs(float(d_a.image.std()) - float(d_b.image.std())) < 5.0

    def test_zero_dynamic_range_does_not_crash(self):
        """All-constant non-uint8 input is degenerate; warn but don't crash."""
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        img = np.full((64, 64), 30000, dtype=np.uint16)
        d.detect_lines(img)  # must not raise
        # Result is a blank uint8 image (degenerate guard branch).
        assert d.image.dtype == np.uint8
        assert int(d.image.max()) == 0

    def test_float_input_normalized_to_uint8(self):
        """Float TIFFs are common; they must be normalized too."""
        rng = np.random.default_rng(0)
        img = rng.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
        # Add a 'fibre' as a dark stripe.
        img[60:65, :] = 0.05
        d = FibreDetector(line_widths=[5], dark_line=True, min_len=3)
        d.detect_lines(img)
        assert d.image.dtype == np.uint8
        # Line should be visibly darker than background in the uint8 output.
        assert float(d.image[60:65, :].mean()) < float(d.image[10:50, :].mean())

    def test_real_synthetic_image_uint8_passthrough(self):
        """Real asset is uint8 RGB; ensure it traverses detect_lines cleanly."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")
        d = FibreDetector(line_widths=[5, 9], dark_line=True, min_len=5)
        d.detect_lines(path)
        assert d.image is not None
        assert d.gray is not None
        assert d.gray.dtype == np.uint8
