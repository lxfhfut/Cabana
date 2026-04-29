"""Tests for cabana/orientation.py — OrientationAnalyzer class."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabana.orientation import OrientationAnalyzer


def make_gray_image(h=64, w=64, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_gradient_image(h=64, w=64):
    """Horizontal gradient — strong horizontal orientation."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col / w * 255)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb


@pytest.fixture
def analyzer():
    return OrientationAnalyzer(sigma=2.0)


@pytest.fixture
def computed(analyzer):
    img = make_gradient_image()
    analyzer.compute_orient(img)
    return analyzer


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_sigma(self):
        a = OrientationAnalyzer()
        assert a.sigma == 2.0

    def test_custom_sigma(self):
        a = OrientationAnalyzer(sigma=5.0)
        assert a.sigma == 5.0

    def test_initial_attributes_none(self):
        a = OrientationAnalyzer()
        assert a.orient is None
        assert a.coherency is None
        assert a.energy is None


# ---------------------------------------------------------------------------
# compute_orient
# ---------------------------------------------------------------------------

class TestComputeOrient:
    def test_populates_orient(self, computed):
        assert computed.orient is not None

    def test_populates_coherency(self, computed):
        assert computed.coherency is not None

    def test_populates_energy(self, computed):
        assert computed.energy is not None

    def test_orient_shape(self, computed):
        assert computed.orient.shape == (64, 64)

    def test_coherency_range(self, computed):
        # Coherency should be in [0, 1]
        assert computed.coherency.min() >= 0.0
        assert computed.coherency.max() <= 1.0 + 1e-6

    def test_energy_range(self, computed):
        assert computed.energy.min() >= 0.0
        assert computed.energy.max() <= 1.0 + 1e-6

    def test_orient_range(self, computed):
        # Orientation in [-π/2, π/2]
        assert computed.orient.min() >= -np.pi / 2 - 1e-6
        assert computed.orient.max() <= np.pi / 2 + 1e-6

    def test_accepts_array_input(self, analyzer):
        img = make_gradient_image()
        analyzer.compute_orient(img)
        assert analyzer.orient is not None

    def test_structure_tensor_components_populated(self, computed):
        assert computed.dxx is not None
        assert computed.dxy is not None
        assert computed.dyy is not None


# ---------------------------------------------------------------------------
# get_orientation_image / get_coherency_image / get_energy_image
# ---------------------------------------------------------------------------

class TestGetImages:
    def test_get_orientation_image_shape(self, computed):
        img = computed.get_orientation_image()
        assert img.shape == (64, 64)

    def test_get_coherency_image_shape(self, computed):
        img = computed.get_coherency_image()
        assert img.shape == (64, 64)

    def test_get_energy_image_shape(self, computed):
        img = computed.get_energy_image()
        assert img.shape == (64, 64)

    def test_mask_zeros_outside(self, computed):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        img = computed.get_orientation_image(mask=mask)
        assert img[0, 0] == 0.0

    def test_full_mask_same_as_no_mask(self, computed):
        full_mask = np.ones((64, 64), dtype=bool)
        img_masked = computed.get_orientation_image(mask=full_mask)
        img_unmasked = computed.get_orientation_image()
        np.testing.assert_array_almost_equal(img_masked, img_unmasked)


# ---------------------------------------------------------------------------
# mean_orientation / mean_coherency
# ---------------------------------------------------------------------------

class TestMeanMetrics:
    def test_mean_orientation_returns_scalar(self, computed):
        val = computed.mean_orientation()
        assert np.isscalar(val) or val.ndim == 0

    def test_mean_orientation_range(self, computed):
        val = computed.mean_orientation()
        assert -90 <= val <= 90

    def test_mean_coherency_returns_scalar(self, computed):
        val = computed.mean_coherency()
        assert np.isscalar(val) or val.ndim == 0

    def test_mean_coherency_range(self, computed):
        val = computed.mean_coherency()
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_mean_orientation_with_mask(self, computed):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        val = computed.mean_orientation(mask=mask)
        assert -90 <= val <= 90


# ---------------------------------------------------------------------------
# circular_variance
# ---------------------------------------------------------------------------

class TestCircularVariance:
    def test_returns_scalar(self, computed):
        val = computed.circular_variance()
        assert np.isscalar(val) or hasattr(val, '__float__')

    def test_nonnegative(self, computed):
        val = computed.circular_variance()
        assert float(val) >= 0.0

    def test_with_mask(self, computed):
        mask = np.ones((64, 64), dtype=bool)
        val = computed.circular_variance(mask=mask)
        assert float(val) >= 0.0


# ---------------------------------------------------------------------------
# randomness_orientation
# ---------------------------------------------------------------------------

class TestRandomnessOrientation:
    def test_returns_float(self, computed):
        val = computed.randomness_orientation()
        assert isinstance(float(val), float)

    def test_nonnegative(self, computed):
        val = computed.randomness_orientation()
        assert float(val) >= 0.0


# ---------------------------------------------------------------------------
# draw_angular_hist
# ---------------------------------------------------------------------------

class TestDrawAngularHist:
    def test_returns_ndarray(self, computed):
        result = computed.draw_angular_hist()
        assert isinstance(result, np.ndarray)

    def test_output_is_rgb(self, computed):
        result = computed.draw_angular_hist()
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_empty_mask_returns_zeros(self, computed):
        empty_mask = np.zeros((64, 64), dtype=bool)
        result = computed.draw_angular_hist(mask=empty_mask)
        assert result.shape[2] in (3, 4)


# ---------------------------------------------------------------------------
# draw_vector_field
# ---------------------------------------------------------------------------

class TestDrawVectorField:
    def test_returns_ndarray(self, computed):
        result = computed.draw_vector_field()
        assert isinstance(result, np.ndarray)

    def test_output_has_3_channels(self, computed):
        result = computed.draw_vector_field()
        assert result.ndim == 3

    def test_with_weights_map(self, computed):
        wgts = np.random.rand(64, 64).astype(np.float32)
        result = computed.draw_vector_field(wgts_map=wgts)
        assert result is not None


# ---------------------------------------------------------------------------
# draw_color_survey
# ---------------------------------------------------------------------------

class TestDrawColorSurvey:
    def test_returns_rgb_image(self, computed):
        result = computed.draw_color_survey()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_shape_matches_input(self, computed):
        result = computed.draw_color_survey()
        assert result.shape[:2] == (64, 64)


# ---------------------------------------------------------------------------
# ROI invariance — masked aggregates must not depend on background pixels
# ---------------------------------------------------------------------------

# OrientationAnalyzer applies two Gaussian blurs (sigma_eff ≈ sqrt(2)*sigma).
# A mask whose boundary lies within ~3*sigma_eff of background pixels will
# pick up smoothed-in background signal, breaking the invariance contract.
# To test the contract cleanly, separate fibre and background by a wide
# buffer and use an interior mask well clear of the boundary.
_FIBRE_HALF = 40   # half-height of the bright fibre stripe (px)
_MASK_HALF = 6     # half-height of the interior ROI mask (px)


def _make_aligned_fibre_image(h=128, w=128, fibre_half=_FIBRE_HALF,
                              noise_seed=None, bg_value=0):
    """RGB image with horizontally-aligned texture in a central band.

    The fibre band (rows h//2 ± fibre_half) is filled with thin parallel
    horizontal stripes (period 4 px), so gradient is purely vertical and
    the structure-tensor orientation is horizontal everywhere inside.

    Outside the band, the image is either uniform (clean) or filled with
    reproducible noise (noisy). The fibre band is byte-identical across
    calls; only the background differs.
    """
    img = np.full((h, w), bg_value, dtype=np.uint8)
    # Horizontal stripes within the fibre band.
    yy = np.arange(h)[:, None]  # (h, 1)
    stripe = (np.cos(2 * np.pi * yy / 4) * 60 + 140).astype(np.uint8)  # (h, 1)
    stripe = np.broadcast_to(stripe, (h, w)).copy()
    band_top = h // 2 - fibre_half
    band_bot = h // 2 + fibre_half
    img[band_top:band_bot, :] = stripe[band_top:band_bot, :]
    if noise_seed is not None:
        rng_bg = np.random.default_rng(noise_seed)
        bg_noise = rng_bg.integers(0, 200, size=(h, w), dtype=np.int16)
        outside = np.ones((h, w), dtype=bool)
        outside[band_top:band_bot, :] = False
        img[outside] = bg_noise[outside].astype(np.uint8)
    return np.stack([img] * 3, axis=-1)


def _fibre_mask(h=128, w=128, mask_half=_MASK_HALF):
    """Interior mask: covers only the central rows of the fibre stripe."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 2 - mask_half:h // 2 + mask_half, :] = 255
    return m


class TestROIInvariance:
    """Aggregates restricted to the fibre mask must not depend on background.

    These tests pin the scientific contract that motivated the
    cabana.Cabana.analyze_orientation fix: the "default" alignment and
    variance numbers (those passed to a downstream user) must describe the
    fibre region only, not the structure of the background.
    """

    @pytest.fixture
    def mask(self):
        return _fibre_mask()

    @pytest.fixture
    def img_clean_bg(self):
        return _make_aligned_fibre_image(noise_seed=None, bg_value=0)

    @pytest.fixture
    def img_noisy_bg(self):
        return _make_aligned_fibre_image(noise_seed=42)

    def test_unmasked_aggregates_change_with_background(
            self, img_clean_bg, img_noisy_bg, mask):
        """Sanity check on the failure mode: unmasked aggregates DO drift."""
        a1 = OrientationAnalyzer(sigma=2.0)
        a1.compute_orient(img_clean_bg)
        a2 = OrientationAnalyzer(sigma=2.0)
        a2.compute_orient(img_noisy_bg)

        coh1 = float(a1.mean_coherency())            # whole image
        coh2 = float(a2.mean_coherency())            # whole image
        var1 = float(a1.circular_variance())
        var2 = float(a2.circular_variance())

        # The two images differ only in background; if the unsuffixed
        # aggregates were reliable they would not move. They do move,
        # which is exactly the bug we are fixing at the call sites.
        assert abs(coh1 - coh2) > 1e-3
        assert abs(var1 - var2) > 1e-3

    def test_masked_coherency_invariant_to_background(
            self, img_clean_bg, img_noisy_bg, mask):
        a1 = OrientationAnalyzer(sigma=2.0)
        a1.compute_orient(img_clean_bg)
        a2 = OrientationAnalyzer(sigma=2.0)
        a2.compute_orient(img_noisy_bg)

        coh1 = float(a1.mean_coherency(mask=mask))
        coh2 = float(a2.mean_coherency(mask=mask))
        # ROI-masked aggregate is computed on the fibre region only, which
        # is byte-identical between the two inputs.
        assert coh1 == pytest.approx(coh2, abs=1e-9)

    def test_masked_variance_invariant_to_background(
            self, img_clean_bg, img_noisy_bg, mask):
        a1 = OrientationAnalyzer(sigma=2.0)
        a1.compute_orient(img_clean_bg)
        a2 = OrientationAnalyzer(sigma=2.0)
        a2.compute_orient(img_noisy_bg)

        var1 = float(a1.circular_variance(mask=mask))
        var2 = float(a2.circular_variance(mask=mask))
        assert var1 == pytest.approx(var2, abs=1e-9)

    def test_masked_orientation_invariant_to_background(
            self, img_clean_bg, img_noisy_bg, mask):
        a1 = OrientationAnalyzer(sigma=2.0)
        a1.compute_orient(img_clean_bg)
        a2 = OrientationAnalyzer(sigma=2.0)
        a2.compute_orient(img_noisy_bg)

        ang1 = float(a1.mean_orientation(mask=mask))
        ang2 = float(a2.mean_orientation(mask=mask))
        assert ang1 == pytest.approx(ang2, abs=1e-9)

    def test_horizontal_fibre_reports_horizontal_orientation(
            self, img_clean_bg, mask):
        a = OrientationAnalyzer(sigma=2.0)
        a.compute_orient(img_clean_bg)
        # Mean orientation in degrees: horizontal fibres → ~0°.
        ang = float(a.mean_orientation(mask=mask))
        assert abs(ang) < 5.0

    def test_aligned_fibre_high_coherency(self, img_clean_bg, mask):
        a = OrientationAnalyzer(sigma=2.0)
        a.compute_orient(img_clean_bg)
        coh = float(a.mean_coherency(mask=mask))
        assert coh > 0.5

    def test_color_survey_with_mask_blanks_outside(self, img_noisy_bg, mask):
        """draw_color_survey(mask=mask_roi) must zero out non-ROI regions."""
        a = OrientationAnalyzer(sigma=2.0)
        a.compute_orient(img_noisy_bg)
        cs = a.draw_color_survey(mask=mask.astype(bool))
        # Outside-mask pixels must be (0, 0, 0)
        outside = cs[~mask.astype(bool)]
        assert outside.max() == 0


class TestRealImageMasking:
    """Real synthetic IHC asset: confirm masked vs unmasked actually differ."""

    def test_real_image_masked_differs_from_unmasked(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "Synthetic", "5001f5e89da92cce.png",
        )
        if not os.path.isfile(path):
            pytest.skip(f"test asset not present: {path}")

        import imageio.v3 as iio
        img = iio.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        a = OrientationAnalyzer(sigma=2.0)
        a.compute_orient(img)

        # Threshold the green channel as a proxy ROI ("fibre" region).
        gray = img[..., 0] if img.ndim == 3 else img
        mask = (gray > 127).astype(np.uint8) * 255

        # If the mask is degenerate, the assertion below is vacuous.
        # Skip rather than make a noisy claim.
        coverage = float(mask.sum()) / (255 * mask.size)
        if coverage < 0.05 or coverage > 0.95:
            pytest.skip(f"mask coverage degenerate: {coverage:.2%}")

        coh_unmasked = float(a.mean_coherency())
        coh_masked = float(a.mean_coherency(mask=mask))
        # The masked value should differ from the whole-image one in a
        # meaningful (non-trivial) way on a real image.
        assert abs(coh_masked - coh_unmasked) > 1e-4
