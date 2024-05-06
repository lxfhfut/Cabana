import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'
import cv2
import csv
import imutils
import convcrf
import argparse
import numpy as np
import torch.nn.init
from glob import glob
from tqdm import tqdm
from skimage import measure
import torch.optim as optim
from log import Log
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage.morphology import remove_small_objects, remove_small_holes
from models import BackBone, LightConv3x3
from utils import mean_image, cal_greenness, save_result_video

# For reproductivity
SEED = 0
torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Supervised Semantic Segmentation')
    parser.add_argument('--num_channels', default=48, type=int,
                        help='Number of channels')
    parser.add_argument('--max_iter', default=200, type=int,
                        help='Number of maximum iterations')
    parser.add_argument('--min_labels', default=2, type=int,
                        help='Minimum number of labels')
    parser.add_argument('--hue_value', default=1.0, type=float, help='Hue value of the color of interest')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate')
    parser.add_argument('--sz_filter', default=5, type=int,
                        help='CRF filter size')
    parser.add_argument('--rt', default=0.25, type=float,
                        help='Relative greenness threshold')
    parser.add_argument('--mode', type=str, default="both")
    parser.add_argument('--min_size', default=64, type=int,
                        help='The smallest allowable object size')
    parser.add_argument('--max_size', default=2048, type=int,
                        help='The maximal allowable image size')
    # parser.add_argument('--disabled', default=False, help='disable segmentation')
    parser.add_argument('--save_video', default=False,  action='store_true',
                        help='save intermediate results as video')
    parser.add_argument('--save_frame_interval', default=2, type=int,
                        help='save frame every save_frame_interval iterations')
    parser.add_argument('--roi_dir', type=str, default="./output/ROIs")
    parser.add_argument('--bin_dir', type=str, default="./output/Bins")
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    args, _ = parser.parse_known_args()
    return args


def segment_single_image(args):
    # Ensure the segmentation result for the same image (e.g. with different names) to be the same
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ori_img = cv2.imread(args.input)
    img_name = os.path.splitext(os.path.basename(args.input))[0]
    rotated = False
    if ori_img.shape[0] > ori_img.shape[1]:
        ori_img = cv2.rotate(ori_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = True
    ori_width, ori_height = ori_img.shape[::-1][1:]
    img = imutils.resize(ori_img, width=512)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]
    img = img.transpose(2, 0, 1)
    data = torch.from_numpy(np.array([img.astype('float32') / 255.]))
    img_var = torch.Tensor(img.reshape([1, 3, *img_size]))  # 1, 3, h, w

    config = convcrf.default_conf
    config['filter_size'] = args.sz_filter
    gausscrf = convcrf.GaussCRF(conf=config, shape=img_size, nclasses=args.num_channels, use_gpu=True)

    model = BackBone([LightConv3x3], [2], [args.num_channels // 2, args.num_channels])
    if torch.cuda.is_available():
        data = data.cuda()
        img_var = img_var.cuda()
        gausscrf = gausscrf.cuda()
        model = model.cuda()

    data = Variable(data)
    img_var = Variable(img_var)

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    label_colours = np.random.randint(255, size=(100, 3))
    all_image_labels = []
    all_mean_images = []
    all_absolute_greenness = []
    all_relative_greenness = []
    all_thresholded = []
    pbar = tqdm(range(args.max_iter))
    for batch_idx in pbar:
        optimizer.zero_grad()
        output = model(data)[0]
        unary = output.unsqueeze(0)
        prediction = gausscrf.forward(unary=unary, img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)

        im_target = target.data.cpu().numpy()
        image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")
        num_labels = len(np.unique(im_target))
        if args.save_video and not (batch_idx % args.save_frame_interval):
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(img_size[0], img_size[1], 3).astype("uint8")
            mean_img = mean_image(rgb_image, measure.label(image_labels))
            absolute_greenness, relative_greenness = cal_greenness(mean_img, args.hue_value)
            # greenness = np.multiply(relative_greenness, (absolute_greenness > args.at).astype(np.float64))
            thresholded = 255 * ((relative_greenness > args.rt).astype("uint8"))
            all_mean_images.append(mean_img)
            all_absolute_greenness.append(absolute_greenness)
            all_relative_greenness.append(relative_greenness)
            all_thresholded.append(thresholded)
            all_image_labels.append(im_target_rgb)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        pbar.set_description(
            "Processing {0}/{1}: {2}, {3:.2f}".format(batch_idx, args.max_iter, num_labels, loss.item()))

        if num_labels <= args.min_labels:
            Log.logger.debug("nLabels", num_labels, "reached minLabels", args.min_labels, ": ", args.input)
            break

    # im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    # im_target_rgb = im_target_rgb.reshape(img_size[0], img_size[1], 3).astype("uint8")

    if args.save_video:
        save_result_path = os.path.join(args.bin_dir, img_name + "_result.mp4")
        save_result_video(save_result_path, rgb_image, all_image_labels, all_mean_images,
                          all_absolute_greenness, all_relative_greenness, all_thresholded)
    else:
        labels = measure.label(image_labels)
        mean_img = mean_image(rgb_image, labels)
        absolute_greenness, relative_greenness = cal_greenness(mean_img, args.hue_value)
        thresholded = relative_greenness > args.rt
        thresholded = remove_small_holes(thresholded, args.min_size)
        thresholded = remove_small_objects(thresholded, args.min_size)
        mask = 255 * (thresholded.astype("uint8"))

        # thickness = 2
        # kernel = np.ones((thickness, thickness), np.uint8)
        # eroded_roi = cv2.erode(mask, kernel)
        # mask = eroded_roi.astype("float") / 255.0
        #
        # if args.smooth_edge:
        #     mask = cv2.GaussianBlur(mask, (5, 5), 0)
        #
        mask = cv2.resize(mask, (ori_width, ori_height), cv2.INTER_NEAREST)
        # roi_img = ori_img.copy().astype('float')
        # roi_img[:, :, 0] = np.multiply(roi_img[:, :, 0], mask) + (1 - mask) * 228
        # roi_img[:, :, 1] = np.multiply(roi_img[:, :, 1], mask) + (1 - mask) * 228
        # roi_img[:, :, 2] = np.multiply(roi_img[:, :, 2], mask) + (1 - mask) * 228
        roi_img = generate_rois(ori_img, (mask > 128).astype("uint8")*255)

        # mask = (mask*255).astype("uint8")

        if rotated:
            cv2.imwrite(os.path.join(args.roi_dir, img_name + '_roi.png'), cv2.rotate(roi_img, cv2.ROTATE_90_CLOCKWISE))
            cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mask.png'),
                        (cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE) > 128).astype("uint8")*255)
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_label.png'),
            #             cv2.rotate(cv2.cvtColor(im_target_rgb, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mean.png'),
            #             cv2.rotate(cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_color.png'),
            #             cv2.rotate(relative_greenness*255, cv2.ROTATE_90_CLOCKWISE))
        else:
            cv2.imwrite(os.path.join(args.roi_dir, img_name + '_roi.png'), roi_img)
            cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mask.png'), (mask > 128).astype("uint8")*255)
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_label.png'),
            #             cv2.cvtColor(im_target_rgb, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_mean.png'), cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(args.bin_dir, img_name + '_color.png'), relative_greenness*255)

        return np.sum(mask > 128), np.sum(mask > 128)/ori_width/ori_height


def visualize_fibres(img, mask, result_path, thickness=3, border_color=[255, 255, 0]):
    mask_gray = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_mask = cv2.dilate(mask_gray, kernel)

    border_color = np.array(border_color)
    (x_idx, y_idx) = np.where(dilated_mask == 255)

    img_with_border = img.copy()
    for row, col in zip(list(x_idx), list(y_idx)):
        img_with_border[row, col, :] = border_color
    cv2.imwrite(result_path, img_with_border)
    # cv2.imwrite(result_path[:-11] + '_border.png', img_with_border)
    # fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    # axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # axs[1].imshow(cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB))
    # plt.savefig(result_path, bbox_inches='tight', dpi=300)
    # plt.close()


def generate_rois(img, roi, thickness=3, background_color=[228, 228, 228]):
    if roi.ndim > 2 and roi.shape[2] > 1:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if img.shape[:2] != roi.shape:
        roi = cv2.resize(roi, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    roi = cv2.bitwise_not(roi)
    kernel = np.ones((thickness, thickness), np.uint8)
    eroded_roi = cv2.dilate(roi, kernel, iterations=1)
    (x_idx, y_idx) = np.where(eroded_roi == 255)

    img_roi = img.copy()
    for row, col in zip(list(x_idx), list(y_idx)):
        img_roi[row, col, :] = np.array(background_color)
    return img_roi


if __name__ == "__main__":
    header = ['Image', 'Area', '% Black']
    args = parse_args()
    
    for num_labels in [48]:
        setattr(args, 'num_channels', num_labels)
        dst_folder = '/home/lxfhfut/Dropbox/Garvan/Test_ROI/relative_0.23/' + str(num_labels)
        src_folder = '/home/lxfhfut/Dropbox/Garvan/Compressed images/'
        img_names = glob(os.path.join(src_folder, '*.tif')) \
                    + glob(os.path.join(src_folder, '.tiff')) \
                    + glob(os.path.join(src_folder, '*.TIF')) \
                    + glob(os.path.join(src_folder, '*.TIFF')) \
                    + glob(os.path.join(src_folder, '*.png')) \
                    + glob(os.path.join(src_folder, '*.PNG'))
        for iter_num in [30]:
            setattr(args, 'max_iter', iter_num)
            setattr(args, 'save_dir', os.path.join(dst_folder, str(iter_num)))

            with open(os.path.join(dst_folder, str(iter_num), 'Results_ROI.csv'), 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for img_name in img_names:
                    print('Processing {}'.format(img_name))
                    setattr(args, 'input', img_name)

                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    area, percent_black = segment_single_image(args)
                    data = [os.path.basename(img_name), area, percent_black]
                    writer.writerow(data)

                print('Result has been saved in {}'.format(args.save_dir))

