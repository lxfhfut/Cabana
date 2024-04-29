import os
import time
import xml.etree.ElementTree as ET
from glob import glob
import re
from PIL import Image
import cv2
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import multivariate_normal
import PIL.Image
import PIL.ExifTags

import seaborn as sns
import seaborn_image as isns
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.segmentation import mark_boundaries
from skimage.morphology import dilation, disk


def contains_oversized(img_paths, max_res=2048):
    max_size = max_res * max_res
    for img_path in img_paths:
        image = Image.open(img_path)
        resolution = image.size
        if resolution[0] * resolution[1] > max_size:
            return True
    return False


def normalize(x, pmin=2, pmax=98, axis=None, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    return np.clip(x, 0, 1)


def info_color_map(img, info_map, cbar_label="Length", cmap="PiYG", radius=1):
    '''This function aims to overlay curve or length map onto the original image.
    Some black spots might be observed in the resultant images.
    These are caused by the missing pixels in the length map.'''
    height, width = info_map.shape[:2]

    if np.max(info_map) > 1:
        cbar_ticks = np.arange(np.min(info_map), np.max(info_map), (np.max(info_map) - np.min(info_map)) // 4)
    else:
        cbar_ticks = np.linspace(0, 1, 5)

    # info_map_normalized = (normalize(info_map) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(width/100.0, height/100.0))
    ax = isns.imgplot(info_map, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
                      cbar=False, dx=5, units="um")
    ax.set_xticks([])
    ax.set_yticks([])

    # # get image data
    # time.sleep(1.0)
    # image_data = ax.figure.canvas.get_renderer().tostring_rgb()
    # fig_width, fig_height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    # image_data = np.frombuffer(image_data, dtype=np.uint8).reshape((int(fig_height), int(fig_width), 3))
    # cv2.imwrite(r"C:\Users\lxfhf\OneDrive\Documents\Debug\tmp_img.png", image_data)
    # plt.close()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    user_home_dir = os.path.expanduser("~")
    fig.savefig(join_path(user_home_dir, ".tmp_image.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    image_data = cv2.imread(join_path(os.path.expanduser("~"), ".tmp_image.png"))
    X = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    color_info_map = np.flipud(cv2.resize(X, (width, height)))

    fig, ax = plt.subplots(figsize=(width/100.0, height/100.0))
    ax = isns.imgplot(info_map, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
                      cbar_label=cbar_label,
                      cbar_ticks=cbar_ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_visible(False)

    fig.savefig(join_path(user_home_dir, ".tmp_colorbar.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    image_data = cv2.imread(join_path(user_home_dir, ".tmp_colorbar.png"))
    Y = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    color_info_map[:, :, 0] = dilation(color_info_map[:, :, 0], disk(radius))
    color_info_map[:, :, 1] = dilation(color_info_map[:, :, 1], disk(radius))
    color_info_map[:, :, 2] = dilation(color_info_map[:, :, 2], disk(radius))

    index_pos = np.where(dilation(info_map > 0, disk(radius)) == 0)
    color_info_map[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]

    height_ratio = np.min([Y.shape[0] / X.shape[0], 1])
    Y_height = int(height * height_ratio)
    h, w = Y.shape[:2]
    r = Y_height / float(h)
    dim = (int(w * r), Y_height)
    colorbar_img = cv2.resize(Y, dim)
    cb_h, cb_w = colorbar_img.shape[:2]
    hor_gap_ratio = 0.5

    final_img = np.zeros((height, int(width + 3 * hor_gap_ratio * cb_w), 3), dtype=np.uint8) + 255
    final_img[:height, :width, :] = color_info_map
    final_img[(height - cb_h) // 2:(height - cb_h) // 2 + cb_h,
    width + int(hor_gap_ratio * cb_w):width + int(hor_gap_ratio * cb_w) + cb_w, :] = colorbar_img
    if os.path.exists(join_path(user_home_dir, ".tmp_image.png")):
        os.remove(join_path(user_home_dir, ".tmp_image.png"))
    if os.path.exists(join_path(user_home_dir, ".tmp_colorbar.png")):
        os.remove(join_path(user_home_dir, ".tmp_colorbar.png"))

    return final_img


def mask_color_map(img, mask, rgb_color=[0.224, 1.0, 0.0784], sigma=0.5):
    imarray = mask / np.max(mask)
    eroded = binary_erosion(imarray, iterations=2)

    outlines = imarray - eroded

    # Convolve with a Gaussian to effect a blur.
    blur = gaussian_filter(outlines, sigma)

    # Make binary images into neon green.
    outlines = outlines[:, :, None] * rgb_color
    blur = blur[:, :, None] * rgb_color

    # Combine the images and constraint to [0, 1].
    glow = np.clip(outlines + blur, 0, 1)
    glow = (np.squeeze(glow) * 255).astype(np.uint8)

    index_pos = np.where(cv2.cvtColor(glow, cv2.COLOR_RGB2GRAY) == 0)
    glow[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]

    return glow


def orient_vf(img, orient_map, wgts_map=None, color=(255, 255, 0), thickness=1, size=15, scale=80):
    ny, nx = orient_map.shape[:2]
    xstart = (nx - (nx // size) * size) // 2
    ystart = (ny - (ny // size) * size) // 2

    x_blk_num = len(range(xstart, nx, size))
    y_blk_num = len(range(ystart, ny, size))

    size2 = size * size

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

            dx = np.mean(np.cos(orient_map[top:bot, lft:rht]))
            dy = np.mean(np.sin(orient_map[top:bot, lft:rht]))
            blk_stats[(y - ystart) // size, (x - xstart) // size, 2] = dy
            blk_stats[(y - ystart) // size, (x - xstart) // size, 3] = dx

            if wgts_map is not None:
                blk_wgts[(y - ystart) // size, (x - xstart) // size] = np.max(wgts_map[top:bot, lft:rht])

    min_val = np.min(blk_wgts)
    max_val = np.max(blk_wgts)

    if min_val != max_val:
        blk_wgts = normalize(blk_wgts, pmin=5, pmax=5, axis=[0, 1])

    vf = np.zeros((ny, nx, 3), dtype=np.uint8)

    for blk_yi in range(y_blk_num):
        for blk_xi in range(x_blk_num):
            r = blk_wgts[blk_yi, blk_xi] * scale / 100.0 * size * 0.5
            y1 = int(blk_stats[blk_yi, blk_xi, 0] + size // 2 - r * blk_stats[blk_yi, blk_xi, 2])
            x1 = int(blk_stats[blk_yi, blk_xi, 1] + size // 2 + r * blk_stats[blk_yi, blk_xi, 3])
            y2 = int(blk_stats[blk_yi, blk_xi, 0] + size // 2 + r * blk_stats[blk_yi, blk_xi, 2])
            x2 = int(blk_stats[blk_yi, blk_xi, 1] + size // 2 - r * blk_stats[blk_yi, blk_xi, 3])
            vf = cv2.line(vf, (x1, y1), (x2, y2), color, thickness)
    # index_pos = np.where(((vf[:, :, 0] != color[0]) | (vf[:, :, 1] != color[1]) | (vf[:, :, 2] != color[2])))
    # vf[index_pos[0], index_pos[1], :] = img[index_pos[0], index_pos[1], :]
    return cv2.addWeighted(img, 0.7, vf, 0.7, 10)
    # return vf


def width_color_map(img, width_img, mask_img, width_color=[0, 255, 255], mask_color=[255, 255, 0]):

    contour_img = img.copy()
    index_pos = np.where((mask_img[:, :, 0] == 255))
    contour_img[index_pos[0], index_pos[1], :] = mask_color

    contour_img = (mark_boundaries(contour_img, (width_img[:, :, 0] > 128).astype(np.uint8),
                                   color=[i/255 if i > 1 else i for i in width_color]) * 255).astype(np.uint8)
    # contours, _ = cv2.findContours(255-width_img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(contour_img, contours, -1, width_color, 1)

    return contour_img
    # width_mask_img = (mark_boundaries(width_img, (width_img[:, :, 0] > 128).astype(np.uint8)) * 255).astype(np.uint8)
    #
    # index_pos = np.where((width_img[:, :, 0] == 255))
    # width_mask_img[index_pos[0], index_pos[1], :] = width_color
    #
    # index_pos = np.where((mask_img[:, :, 0] == 255) & (mask_img[:, :, 1] == 255) & (mask_img[:, :, 2] == 255))
    # width_mask_img[index_pos[0], index_pos[1], :] = mask_color
    #
    # return cv2.addWeighted(img, 0.7, width_mask_img, 0.6, 10)


def sbs_color_map(img, info_map, save_name, cbar_label="Length", cmap="coolwarm"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    axes[0].imshow(np.flipud(img))
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1] = isns.imgplot(np.flipud(info_map), ax=axes[1],
                           cmap=sns.color_palette(cmap, as_cmap=True),
                           cbar_label=cbar_label)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def sbs_color_survey(img, info_map, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    axes[0].imshow(np.flipud(img))
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(np.flipud(info_map))
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def split2batches(img_paths, max_batch_size=5):
    pixel_res = []
    for img_path in img_paths:
        img_info = PIL.Image.open(img_path)
        img_exif = img_info.getexif()

        if img_exif is None:
            print('Sorry, image has no exif data. Setting to default 1.0.')
            pixel_res.append(1.0)
        else:
            xres = 1.0;
            yres = 1.0;
            found = False
            for key, val in img_exif.items():
                if key in PIL.ExifTags.TAGS:
                    if PIL.ExifTags.TAGS[key] == "XResolution":
                        xres = round(1.0/float(val), 2)
                        found = True
                    if PIL.ExifTags.TAGS[key] == "YResolution":
                        yres = round(1.0/float(val), 2)
                        found = True
            if found:
                if xres != yres:
                    print('Warning: XResolution and YResolution in metadata are different! Using XResolution...')
                pixel_res.append(xres)
            else:
                print('Warning: No pixel resolution available in metadata! Setting to default 1.0.')
                pixel_res.append(1.0)
    assert len(pixel_res) == len(img_paths)

    # sort image path based on the pixel resolution
    img_paths = [x for _, x in sorted(zip(pixel_res, img_paths))]
    pixel_res = [y for y, _ in sorted(zip(pixel_res, img_paths))]
    path_batches = []
    res_batches = []

    pres_value = pixel_res[0]
    path_batch = [img_paths[0]]

    for res, img_path in zip(pixel_res[1:], img_paths[1:]):
        if pres_value == res:
            if len(path_batch) < max_batch_size:
                path_batch.append(img_path)
            else:
                path_batches.append(path_batch)
                res_batches.append(pres_value)
                path_batch = [img_path]
        else:
            path_batches.append(path_batch)
            res_batches.append(pres_value)
            path_batch = [img_path]
            pres_value = res

    if len(path_batch) > 0:
        path_batches.append(path_batch)
        res_batches.append(pres_value)

    # double check
    # for i in range(len(path_batches)):
    #     for j in range(len(path_batches[i])):
    #         old_img_res = pixel_res[img_paths.index(path_batches[i][j])]
    #         if old_img_res != res_batches[i]:
    #             print("Image resolution inconsistent! Aborting...")
    #             os._exit(1)
    return path_batches, res_batches


def join_path(*args):
    return os.path.join(*args).replace("\\", "/")


def create_folder(folder, overwrite=True):
    if os.path.exists(folder):
        if overwrite:
            shutil.rmtree(folder)
            os.mkdir(folder)
    else:
        os.makedirs(folder)


def mean_image(image, labels):
    im_rp = image.reshape(-1, image.shape[2])
    labels_1d = np.reshape(labels, -1)
    uni = np.unique(labels_1d)
    uu = np.zeros(im_rp.shape)
    for i in uni:
        loc = np.where(labels_1d == i)[0]
        mm = np.mean(im_rp[loc, :], axis=0)
        uu[loc, :] = mm
    return np.reshape(uu, [image.shape[0], image.shape[1], image.shape[2]]).astype('uint8')


def cal_greenness(rgb_image, hue=1.0):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 0] = hsv[:, :, 0] / 180.0
    hsv[:, :, 1] = hsv[:, :, 1] / 255.0
    hsv[:, :, 2] = hsv[:, :, 2] / 255.0

    # mu = np.array([60.0 / 180.0, 160.0 / 255.0, 200.0 / 255.0])
    if np.mean(hsv[:, :, :2]) == 0:  # grayscale
        mu = np.array([0, 0, 0.99])
    else:
        mu = np.array([hue, 160.0 / 255.0, 200.0 / 255.0])  # this is the mean value previously used for color images
    # mu = np.array([hue, sat, val])
    sigma = np.array([.1, .3, .5])
    covariance = np.diag(sigma ** 2)

    rv = multivariate_normal(mean=mu, cov=covariance)
    z = rv.pdf(hsv)
    ref = rv.pdf(mu)
    absolute_greenness = z/ref
    relative_greenness = (z - np.min(z)) / (np.max(z) - np.min(z) + np.finfo(float).eps)

    return absolute_greenness, relative_greenness


def crop_img_from_center(img, crop_size=(512, 512)):
    assert(img.shape[0] >= crop_size[0])
    assert(img.shape[1] >= crop_size[1])
    assert(len(img.shape)==2 or len(img.shape)==3)
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - crop_size[1] // 2
    y = ch - crop_size[0] // 2
    if len(img.shape) == 3:
        return img[y:y + crop_size[0], x:x + crop_size[1], :]
    else:
        return img[y:y + crop_size[0], x:x + crop_size[1]]


def crop_img_from_center(img, width=512):
    assert(img.shape[1] >= width)
    assert (len(img.shape) == 2 or len(img.shape) == 3)
    height = img.shape[0] * width // img.shape[1]
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - width // 2
    y = ch - height // 2
    if len(img.shape) == 3:
        return img[y:y + height, x:x + width, :]
    else:
        return img[y:y + height, x:x + width]


def save_result_img(save_path, rgb_img, img_labels, mean_img, absolute_greenness, relative_greenness, thresholded):
    # cv2.imwrite(os.path.join(save_path, 'rgb_4.png'), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    # # cv2.imwrite(os.path.join(save_path, 'labels_3.png'), cv2.cvtColor(img_labels, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(save_path, 'mean_4.png'), cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, 'K684.vsi - 40x_BF_01Annotation (Polygon) (Malignant area)_0.png'), thresholded)
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(2, 3, 1)
    # ax.set_title('Original image')
    # plt.axis('off')
    # plt.imshow(rgb_img)

    # ax = fig.add_subplot(2, 3, 2)
    # ax.set_title('Semantic segmentation')
    # plt.axis('off')
    # plt.imshow(img_labels)

    # ax = fig.add_subplot(2, 3, 3)
    # ax.set_title('Mean image')
    # plt.axis('off')
    # plt.imshow(mean_img)

    # ax = fig.add_subplot(2, 3, 4)
    # ax.set_title('Binary mask')
    # plt.axis('off')
    # plt.imshow(thresholded, cmap='gray')

    # ax = fig.add_subplot(2, 3, 5)
    # ax.set_title('Relative greenness')
    # plt.axis('off')
    # plt.imshow(relative_greenness, cmap='gray', vmin=0, vmax=1)

    # ax = fig.add_subplot(2, 3, 6)
    # ax.set_title('Absolute greenness')
    # plt.axis('off')
    # plt.imshow(absolute_greenness, cmap='gray', vmin=0, vmax=1)

    # plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.show(block=False)
    # plt.close("all")


def save_result_video(save_path, rgb_img, all_img_labels, all_mean_imgs, all_absolute_greenness, all_relative_greenness, all_masks):
    imgs = []
    fig = plt.figure(figsize=(15, 10))

    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Original image')
        ax1.axis('off')
        ax1.imshow(cv2.resize(rgb_img, (512, 512)))

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Mean image')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3])

        # ax4 = fig.add_subplot(2, 3, 4)
        # ax4.set_title('Binary mask')
        # ax4.axis('off')
        # ax4.imshow(cv2.resize(all_masks[i], (512, 512)), cmap='gray')
        #
        # ax5 = fig.add_subplot(2, 3, 5)
        # ax5.set_title('Relative redness')
        # ax5.axis('off')
        # ax5.imshow(cv2.resize(all_relative_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)
        #
        # ax6 = fig.add_subplot(2, 3, 6)
        # ax6.set_title('Relative greenness')
        # ax6.axis('off')
        # ax6.imshow(cv2.resize(all_absolute_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        # plt.tight_layout()
        # imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path, fps=5)


def save_result_video_old(save_path, rgb_img, gt_mask, all_img_labels, all_mean_imgs, all_greenness, all_masks):
    imgs = []
    fig = plt.figure(figsize=(10, 15))

    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title('Original image')
        ax1.axis('off')
        ax1.imshow(cv2.resize(rgb_img, (512, 512)))

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax5 = fig.add_subplot(2, 3, 3)
        ax5.set_title('Mean image')
        ax5.axis('off')
        ax5.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Ground truth')
        ax4.axis('off')
        ax4.imshow(cv2.resize(gt_mask, (512, 512)), cmap='gray')

        ax3 = fig.add_subplot(2, 3, 5)
        ax3.set_title('Binary mask')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_masks[i], (512, 512)), cmap='gray')

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Greenness')
        ax6.axis('off')
        ax6.imshow(cv2.resize(all_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path)


def color_coded_map(gt, det):
    gt = gt.astype(bool)
    det = det.astype(bool)
    green_area = np.logical_and(det, gt)
    red_area = np.logical_and(det, np.logical_not(gt))
    blue_area = np.logical_and(np.logical_not(det), gt)

    color_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[green_area] = 255
    color_map[:, :, 1] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[red_area] = 255
    color_map[:, :, 2] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[blue_area] = 255
    color_map[:, :, 0] = tmp_map
    return color_map


def sanitize_filename(filename):
    # Define the set of characters to be removed
    forbidden_chars = r"[ ,:?\/*]"

    # Use regular expressions to remove forbidden characters
    sanitized_filename = re.sub(forbidden_chars, '_', filename)

    return sanitized_filename


def export_parameters(param_path, out_file):
    if not os.path.exists(param_path):
        print("{} not exists.".format(param_path))
    else:
        with open(out_file, 'a+') as hf:
            if os.path.basename(param_path).endswith('.txt'):
                str_header = f"\n******{os.path.basename(param_path)}******\n"
                hf.write(str_header)
                with open(param_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        param_pair = line.rstrip().split(",")
                        key = param_pair[0]
                        value = param_pair[1]
                        hf.write(f"{key}:   {value}\n")
                str_footer = '*' * ((len(str_header) - 3) // 2) + "End" + '*' * ((len(str_header) - 3) // 2) + "\n"
                hf.write(str_footer)
            # elif os.path.basename(param_path).endswith('.xml'):
            #     tree = ET.parse(param_path)
            #     root = tree.getroot()
            #
            #     for entry in root.iter('entry'):
            #         key = entry.attrib['key']
            #         text = entry.text.strip()
            #         hf.write(f"{key}: {text}\n")
            # else:
            #     pass



def get_img_paths(folder, image_types=['*.[Tt][Ii][Ff]*', '*.[Pp][Nn][Gg]', '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]']):
    img_paths = []
    for image_type in image_types:
        img_paths.extend(glob(join_path(folder, image_type)))
    return img_paths


def convert_parameters(param_file_in_micros, param_file_in_pixels, ims_res):
    with open(param_file_in_micros, 'r') as rf, open(param_file_in_pixels, 'w+') as wf:
        lines = rf.readlines()
        for line in lines:
            param_pair = line.rstrip().split(",")
            key = param_pair[0]
            value = param_pair[1]
            if key.lower().startswith("dark line"):
                wf.write(line)
            elif key.lower().startswith("contrast saturation"):
                wf.write(line)
            elif key.lower().startswith("min line width"):
                wf.write("Min Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("max line width"):
                wf.write("Max Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("line width step"):
                wf.write("Line Width Step,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("low contrast") or key.lower().startswith("high contrast"):
                wf.write(line)
            elif key.lower().startswith("min curvature window"):
                wf.write("Min Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("max curvature window"):
                wf.write("Max Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("minimum branch length"):
                wf.write("Minimum Branch Length,{:d}\n".format(int(float(value) / ims_res)))
            elif key.lower().startswith("maximum display hdm"):
                wf.write(line)
            elif key.lower().startswith("minimum gap diameter"):
                wf.write("Minimum Gap Diameter,{:d}\n".format(int(float(value) / ims_res)))
            else:
                print('Invalid parameter {}'.format(key))


if __name__ == "__main__":
    path = join_path("C:\\Users\\Dcouments", "TWOMBLI", "")
    print(path)