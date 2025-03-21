import numpy as np
from scipy.stats import circvar, skew, kurtosis
import imageio.v3 as iio
import cv2
from skimage.util import img_as_float, img_as_bool
from skimage.filters import scharr
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from utils import normalize


class OrientationAnalyzer:
    def __init__(self, sigma=2.0):
        self.sigma = sigma
        self.image = None
        self.gray = None
        self.orient = None
        self.coherency = None
        self.energy = None
        self.dxx = None
        self.dxy = None
        self.dyy = None

    def compute_orient(self, image):
        self.image = iio.imread(image) if isinstance(image, str) else image

        # Normalize to uint8 if needed
        if self.image.dtype != np.uint8:
            self.image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if self.image.ndim == 3 else self.image

        gray = self.gray.astype(float)

        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3,	borderType=cv2.BORDER_REFLECT)
        grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3,	borderType=cv2.BORDER_REFLECT)

        dxx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        dxy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        dyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # Ensure energy in [0, 1]
        energy = dxx + dyy
        self.energy = normalize(energy, 2, 98)

        # Ensure orientation is in [-pi/2, pi/2]
        self.orient = 0.5 * np.arctan2(2.0 * dxy, dyy - dxx)
        self.orient[self.orient > np.pi / 2] = self.orient[self.orient > np.pi / 2] - np.pi
        self.orient[self.orient < -np.pi / 2] = self.orient[self.orient < -np.pi / 2] + np.pi

        # Coherency in[0, 1]
        self.coherency = np.sqrt((dyy - dxx) ** 2 + 4.0 * dxy ** 2) / (dxx + dyy + np.finfo(float).eps)

        self.dxx = dxx
        self.dxy = dxy
        self.dyy = dyy

    def get_orientation_image(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        orient_image = np.zeros_like(self.gray, dtype=float)
        orient_image[mask] = self.orient[mask]
        return orient_image

    def get_coherency_image(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        coherency_image = np.zeros_like(self.gray, dtype=float)
        coherency_image[mask] = self.coherency[mask]
        return coherency_image

    def get_energy_image(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        energy_image = np.zeros_like(self.gray, dtype=float)
        energy_image[mask] = self.energy[mask]
        return energy_image

    def mean_orientation(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)

        vxy = np.mean(self.dxy[mask])
        vxx = np.mean(self.dxx[mask])
        vyy = np.mean(self.dyy[mask])
        return np.rad2deg(0.5*np.arctan2(2.0*vxy, vyy-vxx))

    def mean_coherency(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        vxy = np.mean(self.dxy[mask])
        vxx = np.mean(self.dxx[mask])
        vyy = np.mean(self.dyy[mask])
        return np.sqrt((vyy-vxx)**2 + 4.0*vxy**2) / (vxx + vyy + np.finfo(float).eps)

    def circular_variance(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        return circvar(self.orient[mask] + np.pi/2.0, high=np.pi)

    def randomness_orientation(self, bins=180, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)
        hist, _ = np.histogram((self.orient[mask]+np.pi/2)/np.pi*180, bins=bins, range=(0, 180), density=True)

        probabilities = hist[hist > 0] / np.sum(hist[hist > 0])

        # Calculate entropy
        # Create a uniform distribution
        uniform_probabilities = np.full(bins, 1.0 / bins)

        kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probabilities))

        return 1.0 / (np.sqrt(kl_divergence) + 1.0 + np.finfo(float).eps)

    def draw_angular_hist(self, N=8, mask=None):
        mask = np.ones_like(self.gray, dtype=bool) if mask is None else img_as_bool(mask)
        if mask.sum() == 0:
            return np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        fig = Figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(polar=True)

        # draw right half in [-pi/2, pi/2]
        angles = self.orient[mask]
        distribution = np.histogram(angles, bins=N, range=(-0.5*np.pi, 0.5*np.pi), density=True)[0]
        theta = (np.arange(N) + 0.5) * np.pi / N - np.pi/2.0
        width = np.pi / N  # Width of bars
        colors = plt.cm.hsv((theta+np.pi/2.0) / np.pi)

        # draw symmetric half in[pi/2, 1.5*pi]
        ax.bar(theta, distribution, width=width, color=colors)
        distribution = np.histogram(angles+np.pi, bins=N, range=(0.5 * np.pi, 1.5 * np.pi), density=True)[0]
        theta = (np.arange(N) + 0.5) * np.pi / N + np.pi / 2.0
        width = np.pi / N  # Width of bars
        colors = plt.cm.hsv((theta - np.pi / 2.0) / np.pi)
        ax.bar(theta, distribution, width=width, color=colors)

        ax.set_yticklabels([])
        ax.set_xticks([i/4.0*np.pi for i in range(8)])
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                            r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'])
        ax.tick_params(labelsize=8)
        ax.set_title('Angular Distribution', pad=-5, fontsize=8)
        # plt.pause(0.1)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        return X[..., :3]

    def draw_vector_field(self, wgts_map=None, color=(255, 255, 0), thickness=1, size=15, scale=80):
        ny, nx = self.orient.shape[:2]
        xstart = (nx - (nx // size) * size) // 2
        ystart = (ny - (ny // size) * size) // 2

        x_blk_num = len(range(xstart, nx, size))
        y_blk_num = len(range(ystart, ny, size))

        blk_stats = np.zeros((y_blk_num, x_blk_num, 4))
        blk_wgts = np.ones((y_blk_num, x_blk_num))

        for y in range(ystart, ny, size):
            for x in range(xstart, nx, size):
                blk_stats[(y - ystart) // size, (x - xstart) // size, 0] = y
                blk_stats[(y - ystart) // size, (x - xstart) // size, 1] = x

                top = y - size // 2 if y - size // 2 >= 0 else 0
                bot = y + size // 2 if y + size // 2 <= ny else ny
                lft = x - size // 2 if x - size // 2 >= 0 else 0
                rht = x + size // 2 if x + size // 2 <= nx else nx

                dx = np.mean(np.cos(self.orient[top:bot, lft:rht]))
                dy = np.mean(np.sin(self.orient[top:bot, lft:rht]))
                blk_stats[(y - ystart) // size, (x - xstart) // size, 2] = dy
                blk_stats[(y - ystart) // size, (x - xstart) // size, 3] = dx

                if wgts_map is not None:
                    blk_wgts[(y - ystart) // size, (x - xstart) // size] = np.max(
                        wgts_map[top:bot, lft:rht]
                    )

        min_val = np.min(blk_wgts)
        max_val = np.max(blk_wgts)

        if min_val != max_val:
            blk_wgts = normalize(blk_wgts, pmin=5, pmax=5, axis=[0, 1])

        vf = np.zeros((ny, nx, 3), dtype=np.uint8)

        for blk_yi in range(y_blk_num):
            for blk_xi in range(x_blk_num):
                r = blk_wgts[blk_yi, blk_xi] * scale / 100.0 * size * 0.5
                y1 = int(
                    blk_stats[blk_yi, blk_xi, 0]
                    - r * blk_stats[blk_yi, blk_xi, 2]
                )
                x1 = int(
                    blk_stats[blk_yi, blk_xi, 1]
                    + r * blk_stats[blk_yi, blk_xi, 3]
                )
                y2 = int(
                    blk_stats[blk_yi, blk_xi, 0]
                    + r * blk_stats[blk_yi, blk_xi, 2]
                )
                x2 = int(
                    blk_stats[blk_yi, blk_xi, 1]
                    - r * blk_stats[blk_yi, blk_xi, 3]
                )
                vf = cv2.line(vf, (x1, y1), (x2, y2), color, thickness)

        return cv2.addWeighted(np.atleast_3d(self.image), 0.7, vf, 0.7, 20)

    def draw_color_survey(self, mask=None):
        mask = np.ones_like(self.gray, dtype=bool)if mask is None else img_as_bool(mask)

        # Normalize orientation to [0, 1] then scale to [0, 179] for hue
        hue = (((self.orient + np.pi/2) / np.pi) * 179).astype(np.uint8)

        # Scale coherency and energy to [0, 255] for saturation and value
        saturation = (self.coherency * 255).astype(np.uint8)
        value = (self.energy * 255).astype(np.uint8)
        hue[~mask] = 0
        saturation[~mask] = 0
        value[~mask] = 0

        # Stack the channels to create an HSV image
        hsv_image = np.stack((hue, saturation, value), axis=-1)

        # Convert HSV to RGB for display or saving using OpenCV
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return rgb_image


if __name__ == "__main__":

    dir_analyzer = OrientationAnalyzer(3.0)
    dir_analyzer.compute_orient(
        '/Users/lxfhfut/Dropbox/Garvan/Cabana/539.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif')
    print(dir_analyzer.circular_variance())
    print(kurtosis(dir_analyzer.orient.flatten()))
    circular_hist = dir_analyzer.draw_angular_hist()
    fig, ax = plt.subplots()
    ax.imshow(circular_hist)
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.show()
    # dir_analyzer.draw_angular_hist()
    # fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # axes[0, 0].imshow(rgb)
    # axes[0, 1].imshow(energy, cmap='gray')
    # axes[0, 2].imshow(orient, cmap='gray')
    # axes[1, 0].imshow(coherency, cmap='gray')
    # axes[1, 1].imshow(vector_field)
    # axes[1, 2].imshow(color_survey)
    # plt.show()