"""I/O helpers: paths, folders, image-path globs, batch splitting, parameter
file conversion. Extracted from the monolithic ``cabana.utils`` for
readability. ``cabana.utils`` re-exports the public names below so existing
``from .utils import join_path`` style imports continue to work.
"""

import os
import re
import shutil
from glob import glob
from PIL import Image, ExifTags


read_bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % (
    "\033[0;34m", "\033[0;34m", "\033[0;34m")


def join_path(*args):
    return os.path.join(*args).replace("\\", "/")


def create_folder(folder, overwrite=True):
    if os.path.exists(folder):
        if overwrite:
            shutil.rmtree(folder)
            os.mkdir(folder)
    else:
        os.makedirs(folder)


def get_img_paths(folder,
                  image_types=('*.[Tt][Ii][Ff]*', '*.[Pp][Nn][Gg]',
                               '*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]')):
    img_paths = []
    for image_type in image_types:
        img_paths.extend(glob(join_path(folder, image_type)))
    return img_paths


def sanitize_filename(filename):
    forbidden_chars = r"[ ,:?\/*]"
    return re.sub(forbidden_chars, '_', filename)


def contains_oversized(img_paths, max_res=2048):
    max_size = max_res * max_res
    for img_path in img_paths:
        image = Image.open(img_path)
        resolution = image.size
        if resolution[0] * resolution[1] > max_size:
            return True
    return False


def split2batches(img_paths, max_batch_size=5):
    pixel_res = []
    for img_path in img_paths:
        img_info = Image.open(img_path)
        img_exif = img_info.getexif()

        if img_exif is None:
            print('Sorry, image has no exif data. Setting to default 1.0.')
            pixel_res.append(1.0)
        else:
            xres, yres = 1.0, 1.0
            found = False
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    if ExifTags.TAGS[key] == "XResolution":
                        xres = round(1.0 / float(val), 2)
                        found = True
                    if ExifTags.TAGS[key] == "YResolution":
                        yres = round(1.0 / float(val), 2)
                        found = True
            if found:
                if abs(xres - yres) > 0.01:
                    print('Warning: XResolution and YResolution in metadata are different! Using XResolution...')
                pixel_res.append(xres)
            else:
                print('Warning: No pixel resolution available in metadata! Setting to default 1.0.')
                pixel_res.append(1.0)
    assert len(pixel_res) == len(img_paths)

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

    return path_batches, res_batches


def export_parameters(param_path, out_file):
    if not os.path.exists(param_path):
        print("{} not exists.".format(param_path))
        return
    with open(out_file, 'a+') as hf:
        if os.path.basename(param_path).endswith('.txt'):
            str_header = f"\n******{os.path.basename(param_path)}******\n"
            hf.write(str_header)
            with open(param_path) as f:
                for line in f:
                    key, _, value = line.rstrip().partition(",")
                    hf.write(f"{key}:   {value}\n")
            str_footer = '*' * ((len(str_header) - 3) // 2) + "End" + '*' * ((len(str_header) - 3) // 2) + "\n"
            hf.write(str_footer)


def convert_parameters(param_file_in_micros, param_file_in_pixels, ims_res):
    with open(param_file_in_micros, 'r') as rf, open(param_file_in_pixels, 'w+') as wf:
        for line in rf:
            key, _, value = line.rstrip().partition(",")
            kl = key.lower()
            if kl.startswith("dark line") or kl.startswith("contrast saturation") \
                    or kl.startswith("low contrast") or kl.startswith("high contrast") \
                    or kl.startswith("maximum display hdm"):
                wf.write(line)
            elif kl.startswith("min line width"):
                wf.write("Min Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("max line width"):
                wf.write("Max Line Width,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("line width step"):
                wf.write("Line Width Step,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("min curvature window"):
                wf.write("Min Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("max curvature window"):
                wf.write("Max Curvature Window,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("minimum branch length"):
                wf.write("Minimum Branch Length,{:d}\n".format(int(float(value) / ims_res)))
            elif kl.startswith("minimum gap diameter"):
                wf.write("Minimum Gap Diameter,{:d}\n".format(int(float(value) / ims_res)))
            else:
                print('Invalid parameter {}'.format(key))
