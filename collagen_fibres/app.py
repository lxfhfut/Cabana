import os
import cv2
import yaml
import torch
import imutils
import convcrf
import argparse
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import imageio.v3 as iio
from pathlib import Path
from skimage import measure
from detector import FibreDetector
from torch.autograd import Variable
from segmenter import generate_rois
from models import BackBone, LightConv3x3
from utils import mean_image, cal_greenness
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import euclidean_distances
from skimage.morphology import remove_small_objects, remove_small_holes

SEED = 0
torch.use_deterministic_algorithms(True)


def hex_to_hue(hex_value):
    # Convert hex to RGB
    r = int(hex_value[1:3], 16) / 255.0
    g = int(hex_value[3:5], 16) / 255.0
    b = int(hex_value[5:7], 16) / 255.0

    # Find maximum and minimum values
    max_val = max(r, g, b)
    min_val = min(r, g, b)

    # Calculate delta
    delta = max_val - min_val

    # Calculate hue
    if delta == 0:
        return 0
    elif max_val == r:
        hue = ((g - b) / delta) % 6
    elif max_val == g:
        hue = ((b - r) / delta) + 2
    else:
        hue = ((r - g) / delta) + 4

    hue /= 6.0

    return hue


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
                        help='Relative color threshold')
    parser.add_argument('--mode', type=str, default="both")
    parser.add_argument('--min_size', default=64, type=int,
                        help='The smallest allowable object size')
    parser.add_argument('--max_size', default=2048, type=int,
                        help='The maximal allowable image size')
    parser.add_argument('--input', type=str, help='Input image path', required=False)
    args, _ = parser.parse_known_args()
    return args


def update_parameters(yml_data, params):
    for key, value in params["Segmentation"].items():
        yml_data["Segmentation"][key] = value

    for key, value in params["Detection"].items():
        yml_data["Detection"][key] = value
    return yml_data


def segment_image(ori_img, args, pb):
    # Ensure the segmentation result for the same image (e.g. with different names) to be the same
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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

    gausscrf = convcrf.GaussCRF(conf=config,
                                shape=img_size,
                                nclasses=args.num_channels,
                                use_gpu=torch.cuda.is_available())

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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for batch_idx in range(args.max_iter):
        pb.progress((batch_idx + 1.0)/args.max_iter)
        optimizer.zero_grad()
        output = model(data)[0]
        unary = output.unsqueeze(0)
        prediction = gausscrf.forward(unary=unary, img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)

        im_target = target.data.cpu().numpy()
        image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if len(np.unique(im_target)) <= args.min_labels:
            st.markdown(f"nLabels {num_labels}, reached minLabels {args.min_labels}: {args.input}")
            break

    labels = measure.label(image_labels)
    mean_img = mean_image(rgb_image, labels)
    absolute_greenness, relative_greenness = cal_greenness(mean_img, args.hue_value)
    thresholded = relative_greenness > args.rt
    thresholded = remove_small_holes(thresholded, args.min_size)
    thresholded = remove_small_objects(thresholded, args.min_size)
    mask = cv2.resize(255 * (thresholded.astype("uint8")), (ori_width, ori_height), cv2.INTER_NEAREST)
    roi_img = generate_rois(ori_img, (mask > 128).astype("uint8")*255)
    return roi_img if not rotated else cv2.rotate(roi_img, cv2.ROTATE_90_CLOCKWISE)


def analyze_gaps(img, min_gap_diameter, pb):
    min_gap_radius = min_gap_diameter / 2
    min_dist = int(np.max([1, min_gap_radius]))
    mask = img.copy()

    # set border pixels to zero to avoid partial circles
    mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

    final_circles = []
    downsample_factor = 2
    while True:
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # downsample distance map and upscale detected centers to original image size
        dist_map_downscaled = cv2.resize(dist_map, None, fx=1/downsample_factor, fy=1/downsample_factor)
        centers_downscaled = peak_local_max(dist_map_downscaled, min_distance=min_dist, exclude_border=False)
        centers = centers_downscaled * downsample_factor

        # centers = peak_local_max(dist_map, min_distance=min_dist, exclude_border=False)
        radius = dist_map[centers[:, 0], centers[:, 1]]

        eligible_centers = centers[radius > min_gap_radius, :]
        eligible_radius = radius[radius > min_gap_radius]
        eligible_circles = np.hstack([eligible_centers, eligible_radius[:, None]])

        if len(eligible_circles) == 0:
            break

        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        while len(eligible_circles) > 0:
            if eligible_circles[1:, :].size > 0:
                pw_euclidean_dist = \
                    euclidean_distances(eligible_circles[[0], :2], eligible_circles[1:, :2])[0]
                pw_radius_sum = eligible_circles[0, 2] + eligible_circles[1:, 2]
                neighbor_idx = np.nonzero(pw_euclidean_dist < pw_radius_sum)[0] + 1
                eligible_circles = np.delete(eligible_circles, neighbor_idx, axis=0)

            circle = eligible_circles[0, :]
            result = cv2.circle(result, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 0), -1)
            final_circles.append(eligible_circles[0, :])
            eligible_circles = np.delete(eligible_circles, 0, axis=0)

        mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        pb.progress(np.count_nonzero(mask == 0) / np.prod(mask.shape[:2]))

    final_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for circle in final_circles:
        final_result = cv2.circle(final_result, (int(circle[1]), int(circle[0])),
                                  int(circle[2]), (0, 0, 255), 1)
    return final_result


st.set_page_config(
    page_title='ParamPlay',
    page_icon=':game_die:',
    layout='wide'
)

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

args = parse_args()
yml_data = yaml.safe_load(Path(os.path.join(os.path.dirname(__file__), "default_params.yml")).read_text())
prompt = st.subheader(":blue[Upload an image to try out parameters!]")
with st.sidebar:
    st.title(':red[Parameter Selection]')
    # st.markdown('---')
    # Add an "Open" button to upload the image file
    image_path = st.file_uploader(" ", type=["png", "jpg", "jpeg", "tiff", "tif"])

    tab1, tab2, tab3 = st.tabs([":violet-background[Segmentation]",
                                ":violet-background[Detection]",
                                ":violet-background[Gap Analysis]"])
    with tab1:
        color_cols = st.columns(2)
        hex_color = color_cols[0].color_picker("Color of Interest", "#F90004")
        color_cols[1].markdown(f"Normalized hue: :red[{hex_to_hue(hex_color):.2f}]")
        color_thresh = st.slider("Color Threshold", 0.0, 1.0, 0.2, step=0.05)
        num_labels = st.slider("Number of Labels", 16, 96, 32, step=8)
        max_num_itrs = st.slider("Max Number of Iterations", 10, 60, 30, step=5)
        yml_data["Segmentation"]["Number of Labels"] = num_labels
        yml_data["Segmentation"]["Max Iterations"] = max_num_itrs
        yml_data["Segmentation"]["Color Threshold"] = color_thresh
        yml_data["Segmentation"]["Normalized Hue Value"] = float("{:.2f}".format(hex_to_hue(hex_color)))
        if 'seg_clicked' not in st.session_state:
            st.session_state.seg_clicked = False

        def seg_click_button():
            st.session_state.seg_clicked = True
        tab_cols = st.columns([0.6, 0.4])
        tab_cols[1].button('Segment', on_click=seg_click_button, type="primary")

    with tab2:
        line_width = st.slider("Line Width (pixels)", 1, 15, (3, 5))
        contrast = st.slider("Contrast", 0, 255, (100, 200))
        min_length = st.slider("Minimum Branch Length", 1, 50, 10)

        sidebar_cols = st.columns([0.5, 0.5])
        dark_line = sidebar_cols[0].checkbox("Dark Line", value=True)
        extend = sidebar_cols[1].checkbox("Extend Line")
        correct = sidebar_cols[0].checkbox("Correct Position")
        yml_data["Detection"]["Min Line Width"] = line_width[0]
        yml_data["Detection"]["Max Line Width"] = line_width[1]
        yml_data["Detection"]["Low Contrast"] = contrast[0]
        yml_data["Detection"]["High Contrast"] = contrast[1]
        yml_data["Detection"]["Minimum Branch Length"] = min_length
        yml_data["Detection"]["Dark Line"] = dark_line
        yml_data["Detection"]["Extend Line"] = extend
        yml_data["Detection"]["Correct Position"] = correct
        if 'det_clicked' not in st.session_state:
            st.session_state.det_clicked = False

        def det_click_button():
            st.session_state.det_clicked = True

        sidebar_cols[1].button('Detect', on_click=det_click_button, type="primary")

    with tab3:
        min_gap_diameter = st.slider("Minimum Gap Diameter (pixels)", 5, 100, 20)
        yml_data["Gap Analysis"]["Minimum Gap Diameter"] = min_gap_diameter
        if 'gap_clicked' not in st.session_state:
            st.session_state.gap_clicked = False

        def gap_click_button():
            st.session_state.gap_clicked = True

        gap_cols = st.columns([0.6, 0.4])
        gap_cols[1].button('Analyze', on_click=gap_click_button, type="primary")

right_cols = st.columns([0.8, 0.2])
yaml_content = yaml.dump(yml_data)
right_cols[1].download_button(
    label="Export Parameters",
    data=yaml_content.encode("utf-8"),
    file_name="Parameters.yml",
    mime="text/yaml",
    type="primary"
)

#  Read the uploaded image
if image_path is not None:
    prompt.empty()
    cols = st.columns(2)
    image = iio.imread(image_path)
    cols[0].image(image, clamp=True, caption="Original Image")

    if st.session_state.det_clicked:
        det = FibreDetector(line_widths=line_width,
                            low_contrast=contrast[0],
                            high_contrast=contrast[1],
                            dark_line=dark_line,
                            extend_line=extend,
                            correct_pos=correct,
                            min_len=min_length)

        det.detect_lines(image)
        _, width_image, binary_contours, _ = det.get_results()
        # st.balloons()
        cols[1].image(width_image, clamp=True, caption="Fibre Image", output_format="PNG")

        im = Image.fromarray(binary_contours)
        with BytesIO() as buf:
            im.save(buf, format='PNG')
            data = buf.getvalue()
        st.download_button(label="Download Binary Fibres", data=data, file_name="binary_fibres.png", type="primary")
        st.session_state.det_clicked = False

    if st.session_state.seg_clicked:
        setattr(args, 'num_channels', num_labels)
        setattr(args, 'max_iter', max_num_itrs)
        setattr(args, 'hue_value', hex_to_hue(hex_color))
        setattr(args, 'rt', color_thresh)
        seg_bar = st.progress(0.0, "Segmentation in progress. Please wait.")
        seg_img = segment_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), args, seg_bar)
        seg_bar.empty()
        st.balloons()
        cols[1].image(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB),
                      clamp=True, caption="Segmented Image", output_format="PNG")
        st.session_state.seg_clicked = False

    if st.session_state.gap_clicked:
        if len(np.unique(image)) != 2:
            st.warning('Gap analysis is only supported for binary images. Please upload a binary image!', icon="⚠️")
        else:
            gap_bar = st.progress(0.0, "Gap analysis in progress. Please wait.")
            gap_img = analyze_gaps(image, min_gap_diameter, gap_bar)
            gap_bar.empty()
            # st.balloons()
            cols[1].image(cv2.cvtColor(gap_img, cv2.COLOR_BGR2RGB),
                          clamp=True, caption="Gap Image", output_format="PNG")
        st.session_state.gap_clicked = False


