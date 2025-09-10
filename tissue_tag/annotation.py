import base64
import random
import copy as cp
import logging
from functools import partial
from io import BytesIO
from collections import OrderedDict

import bokeh
import holoviews as hv
import matplotlib.font_manager as fm
import numpy as np
import panel as pn
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageColor
from bokeh.models import FreehandDrawTool, PolyDrawTool
from holoviews.operation import datashader as hd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from packaging import version
from skimage import feature, future
from skimage.draw import polygon, disk
from skimage.future import trainable_segmentation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from scanpy.preprocessing import normalize_total
from tinybrain import downsample_segmentation

from tissue_tag.io import TissueTagAnnotation

hv.extension('bokeh')

# Holoviews/bokeh custom classes and functions
font_path = fm.findfont('DejaVu Sans')


class CustomFreehandDraw(hv.streams.FreehandDraw):
    """
    This custom class adds the ability to customise the icon for the FreeHandDraw tool.
    """

    def __init__(self, empty_value=None, num_objects=0, styles=None, tooltip=None, icon_colour="black", **params):
        self.icon_colour = icon_colour
        super().__init__(empty_value, num_objects, styles, tooltip, **params)


class CustomFreehandDrawCallback(hv.plotting.bokeh.callbacks.PolyDrawCallback):
    """
    This custom class is the corresponding callback for the CustomFreeHandDraw which will render a custom icon for
    the FreeHandDraw tool.
    """

    def initialize(self, plot_id=None):
        plot = self.plot
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        stream = self.streams[0]
        if stream.styles:
            self._create_style_callback(cds, glyph)
        kwargs = {}
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        kwargs['icon'] = create_icon(stream.tooltip[0], stream.icon_colour)
        poly_tool = FreehandDrawTool(
            num_objects=stream.num_objects,
            renderers=[plot.handles['glyph_renderer']],
            **kwargs
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        hv.plotting.bokeh.callbacks.CDSCallback.initialize(self, plot_id)


class CustomPolyDraw(hv.streams.PolyDraw):
    """
    This custom class adds the ability to customise the icon for the PolyDraw tool.
    """

    def __init__(self, empty_value=None, drag=True, num_objects=0, show_vertices=False, vertex_style={}, styles={},
                 tooltip=None, icon_colour="black", **params):
        self.icon_colour = icon_colour
        super().__init__(empty_value, drag, num_objects, show_vertices, vertex_style, styles, tooltip, **params)


class CustomPolyDrawCallback(hv.plotting.bokeh.callbacks.GlyphDrawCallback):
    """
    This custom class is the corresponding callback for the CustomPolyDraw which will render a custom icon for
    the PolyDraw tool.
    """

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        cds = self.plot.handles['cds']
        glyph = self.plot.handles['glyph']
        renderers = [plot.handles['glyph_renderer']]
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.show_vertices:
            vertex_style = dict({'size': 10}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            kwargs['vertex_renderer'] = r1
        if stream.styles:
            self._create_style_callback(cds, glyph)
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        kwargs['icon'] = 'delete'
        poly_tool = PolyDrawTool(
            drag=all(s.drag for s in self.streams), renderers=renderers,
            **kwargs
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)


# Overload the callback from holoviews to use the custom FreeHandDrawCallback class. Probably not safe.
hv.plotting.bokeh.callbacks.Stream._callbacks['bokeh'].update({
    CustomFreehandDraw: CustomFreehandDrawCallback,
    CustomPolyDraw: CustomPolyDrawCallback
})


def to_base64(img):
    """
    Helper function to convert an Image object to base64 string.

    Parameters
    ----------
    img: PIL.Image
        PIL Image object to convert to base64 string.

    Returns
    -------
    str
        Base64 string of the image in PNG format.
    """

    buffered = BytesIO()
    img.save(buffered, format="png")
    data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{data}'


def create_icon(name, colour):
    """
    Helper function to create a custom icon composed on a coloured rectangle with a letter in the middle.

    Parameters
    ----------
    name: str
        Name of the tool.
    colour: str
        Color of the icon. Can be a color name or an RGB tuple.

    Returns
    -------
    PIL.Image
        PIL Image object of the icon.
    """

    font_size = 24
    size = (30, 30)

    if isinstance(colour, str):
        # Create a temporary 1x1 image to convert color name to RGB
        temp_img = Image.new("RGB", (1, 1), colour)
        colour = temp_img.getpixel((0, 0))
    else:
        colour = colour

    # Create a new image with the specified background color
    image = Image.new('RGB', size, color=colour)
    draw = ImageDraw.Draw(image)

    # Calculate font size if not provided
    if font_size is None:
        font_size = int(min(size) * 0.4)

    # Try to use the specified font, fallback to default if not available
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text position to center it
    # For newer Pillow versions use textbbox
    try:
        # Get the bounding box of the text
        left, top, right, bottom = draw.textbbox((0, 0), name, font=font)
        text_width = right - left
        text_height = bottom - top

        # For better vertical centering with both uppercase and lowercase letters
        # Get metrics for a reference that includes both ascenders and descenders
        ref_left, ref_top, ref_right, ref_bottom = draw.textbbox((0, 0), "Ájpq", font=font)
        ref_height = ref_bottom - ref_top

        # Calculate visual center offset
        ascent_ratio = 0.8  # Approximate ratio for visual center (may need adjustment)

        # Calculate horizontal and vertical positions
        text_x = (size[0] - text_width) // 2 - left  # Adjust for left offset
        text_y = (size[1] - ref_height) // 2 + (ref_height * (1 - ascent_ratio)) - top

    except AttributeError:
        # Fall back to textsize for older Pillow versions
        text_width, text_height = draw.textsize(name, font=font)
        # Get the offset for proper alignment
        try:
            offset_x, offset_y = font.getoffset(name)
            # Get metrics for reference string
            ref_width, ref_height = draw.textsize("Ájpq", font=font)

            # Calculate positions with adjustment for visual center
            text_x = (size[0] - text_width) // 2 - offset_x
            text_y = (size[1] - ref_height) // 2 + (ref_height * 0.3) - offset_y
        except (AttributeError, TypeError):
            # Simplest fallback if getoffset is not available
            text_x = (size[0] - text_width) // 2
            text_y = (size[1] - text_height) // 2

    brightness = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]

    # Use white text on dark backgrounds, black text on light backgrounds
    text_color = "white" if brightness < 128 else "black"

    # Draw the letter
    draw.text((text_x, text_y), name, fill=text_color, font=font, stroke_width=0.2, stroke_fill=text_color)

    if version.parse(bokeh.__version__) < version.parse("3.1.0"):
        image = to_base64(image)
    return image


# Annotation functions

def annotator(tissue_tag_annotation, plot_size=1024, invert_y=False, use_datashader=False,
              unassigned_colour="yellow"):
    """
    Interactive annotation tool with line annotations using Panel for switching between morphology and annotation.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y : bool, optional
        invert plot along y axis
    use_datashader : bool, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.
    unassigned_colour : str, optional
        Color for unassigned pixels. Default is "yellow".

    Returns
    -------
    Panel app
        A panel application object with annotator tool.
    """

    logging.getLogger('bokeh.core.validation.check').setLevel(logging.ERROR)

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide annotation map.")
    else:
        tissue_tag_annotation.annotation_map = OrderedDict(tissue_tag_annotation.annotation_map)
        tissue_tag_annotation.annotation_map["unassigned"] = unassigned_colour
        tissue_tag_annotation.annotation_map.move_to_end("unassigned", last=False)

    if tissue_tag_annotation.label_image is None:
        label_image = np.zeros((tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]),
                               dtype=np.uint8)
        tissue_tag_annotation.label_image = label_image
        provided_annotation_map = tissue_tag_annotation.annotation_map.copy()
        tissue_tag_annotation.annotation_map = {'default': '#00000000'}
        annotation = rgb_from_labels(tissue_tag_annotation)
        tissue_tag_annotation.annotation_map = provided_annotation_map
    else:
        annotation = rgb_from_labels(tissue_tag_annotation)

    annotation_c = annotation.astype('uint8').copy()
    if not invert_y:
        annotation_c = np.flip(annotation_c, 0)

    imarray_c = tissue_tag_annotation.image.astype('uint8').copy()
    if not invert_y:
        imarray_c = np.flip(imarray_c, 0)

    update_button = pn.widgets.Button(name='Update', button_type='primary')
    revert_button = pn.widgets.Button(name='Revert', button_type='danger', disabled=True)
    label_opacity = pn.widgets.FloatSlider(name='Label overlay', value=0.5, start=0, end=1, step=0.1)

    def create_images(annotation_c=annotation_c, imarray_c=imarray_c):
        # Create new holoview images
        anno = hv.RGB(annotation_c, bounds=(0, 0, annotation_c.shape[1], annotation_c.shape[0]))
        if use_datashader:
            anno = hd.regrid(anno)
        ds_anno = (anno.
                   options(aspect="equal", frame_height=int(plot_size),
                           frame_width=int(plot_size), alpha=label_opacity.value).
                   apply.opts(alpha=label_opacity.param.value))
        ds_anno.opts(backend_opts={"plot.toolbar_location": "left"})

        img = hv.RGB(imarray_c, bounds=(0, 0, imarray_c.shape[1], imarray_c.shape[0]))
        if use_datashader:
            img = hd.regrid(img)
        ds_img = img.options(aspect="equal", frame_height=int(plot_size), frame_width=int(plot_size))

        return [ds_img, ds_anno]

    plot_list = create_images()

    render_dict = {}
    path_dict = {}
    for key in tissue_tag_annotation.annotation_map.keys():
        path_dict[key] = hv.Path([]).opts(color=tissue_tag_annotation.annotation_map[key], line_width=5, line_alpha=0.7)
        render_dict[key] = CustomFreehandDraw(source=path_dict[key], num_objects=200, tooltip=key,
                                              icon_colour=tissue_tag_annotation.annotation_map[key])

        plot_list.append(path_dict[key])

    tab_object = pn.panel(hd.Overlay(plot_list).collate())
    # Create the tabbed view
    p = pn.Column(pn.Row(label_opacity, update_button, revert_button), tab_object)

    previous_labels = tissue_tag_annotation.label_image.copy()

    def update_annotator(event):
        nonlocal tab_object, previous_labels, revert_button

        if not event:
            return

        tab_object.loading = True
        update_button.disabled = True

        previous_labels = tissue_tag_annotation.label_image.copy()
        updated_labels = tissue_tag_annotation.label_image.copy()
        for idx, a in enumerate(render_dict.keys()):
            if render_dict[a].data['xs']:
                for o in range(len(render_dict[a].data['xs'])):
                    x = np.array(render_dict[a].data['xs'][o]).astype(int)
                    y = np.array(render_dict[a].data['ys'][o]).astype(int)
                    rr, cc = polygon(y, x)
                    inshape = np.where(
                        np.array(tissue_tag_annotation.label_image.shape[0] > rr) & np.array(0 < rr) & np.array(
                            tissue_tag_annotation.label_image.shape[1] > cc) & np.array(
                            0 < cc))[0]
                    updated_labels[rr[inshape], cc[inshape]] = idx + 1

        tissue_tag_annotation.label_image = updated_labels

        annotation = rgb_from_labels(tissue_tag_annotation)
        annotation_c = annotation.astype('uint8').copy()
        if not invert_y:
            annotation_c = np.flip(annotation_c, 0)

        updated_plot_list = create_images(annotation_c, imarray_c)

        plot_list[0] = updated_plot_list[0]
        plot_list[1] = updated_plot_list[1]
        tab_object = pn.panel(hd.Overlay(plot_list).collate())

        p[1] = tab_object
        revert_button.disabled = False
        update_button.disabled = False

    def revert_annotator(event):
        nonlocal tab_object, previous_labels, revert_button

        if not event:
            return

        tab_object.loading = True
        update_button.disabled = True

        tissue_tag_annotation.label_image = previous_labels
        annotation = rgb_from_labels(tissue_tag_annotation)
        annotation_c = annotation.astype('uint8').copy()
        if not invert_y:
            annotation_c = np.flip(annotation_c, 0)

        updated_plot_list = create_images(annotation_c, imarray_c)

        plot_list[0] = updated_plot_list[0]
        plot_list[1] = updated_plot_list[1]
        tab_object = pn.panel(hd.Overlay(plot_list).collate())

        p[1] = tab_object
        revert_button.disabled = True
        update_button.disabled = False

    pn.bind(update_annotator, update_button, watch=True)
    pn.bind(revert_annotator, revert_button, watch=True)

    return p


def rgb_from_labels(tissue_tag_annotation):
    """
    Helper function to generate colored annotation image from label image and annotation map.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with label_image and annotation map.

    Returns
    -------
    numpy.ndarray
        Annotation image.
    """

    labelimage_rgb = np.zeros(
        (tissue_tag_annotation.label_image.shape[0], tissue_tag_annotation.label_image.shape[1], 4))

    colours = list(tissue_tag_annotation.annotation_map.values())
    for c in range(len(colours)):
        color = ImageColor.getcolor(colours[c], "RGBA")
        labelimage_rgb[tissue_tag_annotation.label_image == c + 1, 0:4] = np.array(color)

    return labelimage_rgb.astype('uint8')


def pixel_label_classifier(tissue_tag_annotation, classifier="RandomForest", threshold=None, downsampling_factor=1, plot=True, copy=False):
    """
    A pixel classifier using all RGB channels as features.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    classifier: str, optional
        Algorithm to use for classification. Currently supported algorithms are RandomForest and LogisticRegression from sklearn.
    threshold: float, optional
        Minimum probability score threshold for a pixel to be assigned a label by the random forest classifier.
    downsampling_factor: integer, optional
        Downsampling factor to resize image and label_image prior to running classifier. Default value of 1 means no downsampling is performed.
    plot: bool, optional
        Plot the resulting label_image after running the clasifier.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated label_image based on the classifier prediction if copy is True,
        otherwise None.
    """

    def predict_segmenter_thresholded(features, clf, threshold):
        sh = features.shape
        if features.ndim > 2:
            features = features.reshape((-1, sh[-1]))

        try:
            predicted_labels_probability = clf.predict_proba(features)
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            )
        except ValueError as err:
            if err.args and 'x must consist of vectors of length' in err.args[0]:
                raise ValueError(
                    err.args[0]
                    + '\n'
                    + "Maybe you did not use the same type of features for training the classifier."
                )
            else:
                raise err

        label_classes = np.concatenate((np.array([0]), clf.classes_))
        max_probability_indices = np.argmax(predicted_labels_probability, axis=1) + 1
        no_label_mask = np.max(predicted_labels_probability, axis=1) < threshold
        max_probability_indices[no_label_mask] = 0
        predicted_labels = label_classes[max_probability_indices]

        output = predicted_labels.reshape(sh[:-1])
        return output

    if classifier not in ["RandomForest", "LogisticRegression"]:
        raise ValueError("Classifier is not supported. Currently supported classifiers are RandomForest and LogisticRegression.")

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation

    print("[INFO] Initializing classifier...")
    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1)  # Process all channels together

    image = tissue_tag_annotation.image
    label_image = tissue_tag_annotation.label_image

    if downsampling_factor > 1:
        print("[INFO] Downsampling image and label_image...")
        label_image = downsample_segmentation(label_image, (downsampling_factor, downsampling_factor))[0]
        image = cv2.resize(image, label_image.shape[::-1], interpolation=cv2.INTER_LANCZOS4)
        print(f"[INFO]  - Original resolution: {tissue_tag_annotation.label_image.shape}. Downsampled resolution: {label_image.shape}")

    print("[INFO] Extracting features from all RGB channels...")
    features = features_func(image)  # Extract multiscale features for all channels at once

    del(image)

    print(f"[INFO] Training {classifier} classifier on RGB features...")
    if classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
    elif classifier == "LogisticRegression":
        clf = LogisticRegression(n_jobs=-1, random_state=42)
    clf = trainable_segmentation.fit_segmenter(label_image, features, clf)

    print("[INFO] Predicting labels based on trained classifier...")
    if threshold is not None:
        print(f"[INFO] - Using {threshold} probability threshold for assigning label.")
        predicted_labels = predict_segmenter_thresholded(features, clf, threshold)
    else:
        predicted_labels = trainable_segmentation.predict_segmenter(features, clf)

    if downsampling_factor > 1:
        print("[INFO] Upsampling predicted label_image...")
        predicted_labels = cv2.resize(predicted_labels, tissue_tag_annotation.label_image.shape[::-1], interpolation=cv2.INTER_NEAREST)

    print("[INFO] Final label prediction completed.")
    tissue_tag_annotation.label_image = predicted_labels

    del(label_image)
    del(predicted_labels)
    del(features)
    del(clf)

    if plot:
        print("[INFO] Generating visualization...")
        plot_labels(tissue_tag_annotation, alpha=0.7)
        print("[INFO] Visualization complete.")

    print("[INFO] Classification finished successfully.")

    return tissue_tag_annotation if copy else None


def overlay_labels(im1, im2, alpha=0.8, show=True):
    """
    Helper function to merge 2 images.

    Parameters
    ----------
    im1 : np.array
        1st image.
    im2 : np.array
        2nd image.
    alpha : float, optional
        Blending factor, by default 0.8.
    show : bool, optional
        If to show the merged plot or not, by default True.

    Returns
    -------
    None | numpy.ndarray
        The merged image array if show is False, otherwise None.
    """

    # Generate overlay image
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape, dtype=im1.dtype)
    out_img[:, :, :] = (alpha * im1[:, :, :]) + ((1 - alpha) * im2[:, :, :])
    out_img[:, :, 3] = 255
    if show:
        plt.imshow(out_img, origin='lower')

    return None if show else out_img


def plot_labels(tissue_tag_annotation, alpha=0.8):
    """
    Helper function to plot the annotation labels on the image.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    alpha : float, optional
        Blending factor, by default 0.8.

    Returns
    -------
    None
    """

    annotation = rgb_from_labels(tissue_tag_annotation)
    return overlay_labels(tissue_tag_annotation.image, annotation, alpha, show=True)


def segmenter(tissue_tag_annotation, plot_size=1024, invert_y=False, use_datashader=False,
              annotation_prefix="object"):
    """
    Interactive annotation tool to segment image using Panel to switch between morphology and annotation.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y : bool
        invert plot along y axis
    use_datashader : bool, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.
    annotation_prefix : str, optional
        Prefix for annotations. Default is "object".

    Returns
    -------
    Panel app
        A panel application object with segmenter tool.
    """

    if tissue_tag_annotation.label_image is None:
        label_image = np.zeros((tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]),
                               dtype=np.uint8)
        tissue_tag_annotation.label_image = label_image
        tissue_tag_annotation.annotation_map = OrderedDict({})

    # convert label image to rgb for annotation
    annotation = rgb_from_labels(tissue_tag_annotation)

    annotation_c = annotation.astype('uint8').copy()
    if not invert_y:
        annotation_c = np.flip(annotation_c, 0)

    imarray_c = tissue_tag_annotation.image.astype('uint8').copy()
    if not invert_y:
        imarray_c = np.flip(imarray_c, 0)

    update_button = pn.widgets.Button(name='Update', button_type='primary')
    revert_button = pn.widgets.Button(name='Revert', button_type='danger', disabled=True)
    label_opacity = pn.widgets.FloatSlider(name='Label overlay', value=0.5, start=0, end=1, step=0.1)

    def create_images(annotation_c=annotation_c, imarray_c=imarray_c):
        # Create new holoview images
        anno = hv.RGB(annotation_c, bounds=(0, 0, annotation_c.shape[1], annotation_c.shape[0]))
        if use_datashader:
            anno = hd.regrid(anno)
        ds_anno = (anno.
                   options(aspect="equal", frame_height=int(plot_size),
                           frame_width=int(plot_size), alpha=label_opacity.value).
                   apply.opts(alpha=label_opacity.param.value))
        ds_anno.opts(backend_opts={"plot.toolbar_location": "left"})

        img = hv.RGB(imarray_c, bounds=(0, 0, imarray_c.shape[1], imarray_c.shape[0]))
        if use_datashader:
            img = hd.regrid(img)
        ds_img = img.options(aspect="equal", frame_height=int(plot_size), frame_width=int(plot_size))

        return [ds_img, ds_anno]

    plot_list = create_images()

    path_object = hv.Path([]).opts(line_width=3, line_alpha=0.7)
    draw_object = hv.streams.PolyDraw(source=path_object, show_vertices=True, num_objects=300, drag=True)
    edit_object = hv.streams.PolyEdit(source=path_object, vertex_style={'color': 'red'}, shared=True)
    plot_list.append(path_object)

    erase_path_object = hv.Path([]).opts(line_width=3, line_alpha=1, line_color="black")
    erase_object = CustomPolyDraw(source=erase_path_object, num_objects=300, show_vertices=True, drag=True,
                                  tooltip="Eraser", vertex_style={'color': 'black'})
    edit_erase_object = hv.streams.PolyEdit(source=erase_path_object, shared=True)
    plot_list.append(erase_path_object)

    tab_object = pn.panel(hd.Overlay(plot_list).collate())
    # Create the tabbed view
    p = pn.Column(pn.Row(label_opacity, update_button, revert_button), tab_object)

    previous_label = tissue_tag_annotation.label_image
    previous_annotation_map = tissue_tag_annotation.annotation_map

    def update_segmenter(event):
        nonlocal tab_object, previous_label, previous_annotation_map, revert_button

        if not event:
            return

        tab_object.loading = True
        update_button.disabled = True

        colorpool = ['green', 'cyan', 'brown', 'magenta', 'blue', 'red', 'orange']

        previous_label = tissue_tag_annotation.label_image.copy()
        previous_annotation_map = tissue_tag_annotation.annotation_map.copy()

        existing_object_count = len(tissue_tag_annotation.annotation_map.keys()) + 1
        print(existing_object_count)
        if erase_object.data['xs']:
            print("Erasing")
            for o in range(len(erase_object.data['xs'])):
                x = np.array(erase_object.data['xs'][o]).astype(int)
                y = np.array(erase_object.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = (tissue_tag_annotation.label_image.shape[0] > rr) & (0 < rr) & (
                            tissue_tag_annotation.label_image.shape[1] > cc) & (
                                  0 < cc)  # make sure pixels outside the image are ignored
                tissue_tag_annotation.label_image[rr[inshape], cc[inshape]] = 0

        if draw_object.data['xs']:
            print("Updating")
            for o in range(len(draw_object.data['xs'])):
                x = np.array(draw_object.data['xs'][o]).astype(int)
                y = np.array(draw_object.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = (tissue_tag_annotation.label_image.shape[0] > rr) & (0 < rr) & (
                            tissue_tag_annotation.label_image.shape[1] > cc) & (
                                  0 < cc)  # make sure pixels outside the image are ignored
                tissue_tag_annotation.label_image[rr[inshape], cc[inshape]] = existing_object_count + o
                tissue_tag_annotation.annotation_map[annotation_prefix + '_' + str(existing_object_count + o)] = (
                    random.choice(colorpool))

        annotation = rgb_from_labels(tissue_tag_annotation)
        annotation_c = annotation.astype('uint8').copy()
        if not invert_y:
            annotation_c = np.flip(annotation_c, 0)

        updated_plot_list = create_images(annotation_c, imarray_c)

        plot_list[0] = updated_plot_list[0]
        plot_list[1] = updated_plot_list[1]
        tab_object = pn.panel(hd.Overlay(plot_list).collate())

        p[1] = tab_object
        revert_button.disabled = False
        update_button.disabled = False

    def revert_segmenter(event):
        nonlocal tab_object, previous_label, previous_annotation_map, revert_button

        if not event:
            return

        tab_object.loading = True

        tissue_tag_annotation.label_image = previous_label.copy()
        tissue_tag_annotation.annotation_map = previous_annotation_map.copy()

        annotation = rgb_from_labels(tissue_tag_annotation)
        annotation_c = annotation.astype('uint8').copy()
        if not invert_y:
            annotation_c = np.flip(annotation_c, 0)

        updated_plot_list = create_images(annotation_c, imarray_c)

        plot_list[0] = updated_plot_list[0]
        plot_list[1] = updated_plot_list[1]
        tab_object = pn.panel(hd.Overlay(plot_list).collate())

        p[1] = tab_object
        revert_button.disabled = True
        update_button.disabled = False

    pn.bind(update_segmenter, update_button, watch=True)
    pn.bind(revert_segmenter, revert_button, watch=True)

    return p


def gene_labels_from_adata(adata, gene_markers, tissue_tag_annotation, diameter, override_labels=False,
                           space_every_spots=10, normalize=True, unassigned_colour="yellow", intensity_threshold=230,
                           copy=False):
    """
    Assign labels to training spots based on gene expression from an existing AnnData object.

    Parameters
    ----------
    adata : AnnData
        Pre-loaded AnnData object containing gene expression data.
    gene_markers : dict
        Dictionary mapping annotation to pairs of marker genes and number of spots to assign.
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with image and annotation map.
    diameter : float
        Radius of the spots.
    override_labels : bool
        Remove existing label_image and replace with new labels.
    space_every_spots : int, optional
        Spacing between background spots. Default is 10.
    normalize : bool
        if to normalize gene expression by default parametres calculated by - scanpy.pp.normalize_total()
    unassigned_colour : str, optional
        Color for unassigned labels. Default is "yellow".
    intensity_threshold: int, optional
        Threshold above which pixels are considered background. Default is 230.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated label_image based on the gene expression if copy is True,
        otherwise None.
    """

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation

    if tissue_tag_annotation.label_image is not None:
        print("Label image is not empty.")
        if override_labels:
            # Initialize label image
            print("Will replace with an empty label_image.")
            tissue_tag_annotation.label_image = np.zeros(
                (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)
        else:
            print("Will add new gene labels on top of old label_image.")
    else:  # if the label_image spot is empty then create a blank one
        tissue_tag_annotation.label_image = np.zeros(
            (tissue_tag_annotation.image.shape[0], tissue_tag_annotation.image.shape[1]), dtype=np.uint8)

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")
    else:
        tissue_tag_annotation.annotation_map = OrderedDict(tissue_tag_annotation.annotation_map)
        tissue_tag_annotation.annotation_map["unassigned"] = unassigned_colour
        tissue_tag_annotation.annotation_map.move_to_end("unassigned", last=False)

    # Filter adata to match df indices
    adata = adata[tissue_tag_annotation.positions.index.intersection(adata.obs.index)]
    r = diameter / 2 * tissue_tag_annotation.ppm

    # Extract coordinates
    labels = background_labels_intensity(tissue_tag_annotation.label_image.shape[:2],
                                         imarray=tissue_tag_annotation.image, r=r,
                                         intensity_threshold=intensity_threshold, space_every_spots=space_every_spots,
                                         label=1)
    mask = tissue_tag_annotation.label_image > 0
    labels[mask] = tissue_tag_annotation.label_image[mask]  # add old labels if these are not empty

    if normalize:
        normalize_total(adata)

    # Assign labels based on gene expression
    for marker, gene_list in gene_markers.items():
        # Get the expected color for the marker
        marker_color = tissue_tag_annotation.annotation_map.get(marker, "N/A")
        print(f"🧬 Processing marker: '{marker}' | Color: {marker_color} | Genes: {[gene for gene, _ in gene_list]}")

        combined_gene_indices = []

        for gene, top_n in gene_list:
            GeneIndex = np.where(adata.var_names.str.fullmatch(gene))[0]
            if GeneIndex.size == 0:
                print(f"Warning: Gene {gene} not found in AnnData. Skipping.")
                continue

            GeneData = adata.X[:, GeneIndex].todense().A1  # Flatten to 1D array
            nonzero_indices = np.where(GeneData > 0)[0]

            if len(nonzero_indices) == 0:
                print(f"Warning: No non-zero expression for gene {gene}. Skipping.")
                continue

            # Build a DataFrame to sort and shuffle
            gene_df = pd.DataFrame({
                "barcode": adata.obs.index[nonzero_indices],
                "expression": GeneData[nonzero_indices]
            })

            # Shuffle within expression levels to avoid spatial artifacts
            gene_df = gene_df.groupby("expression", group_keys=False).apply(lambda x: x.sample(frac=1))

            # Now sort by expression descending
            gene_df_sorted = gene_df.sort_values("expression", ascending=False)

            # Take top N
            actual_top_n = min(top_n, len(gene_df_sorted))
            selected_barcodes = gene_df_sorted["barcode"].iloc[:actual_top_n]

            combined_gene_indices.extend(selected_barcodes)

        # Remove duplicates and convert to a set for faster lookups later
        combined_gene_indices = set(combined_gene_indices)

        # Assign labels
        for idx, sub in enumerate(tissue_tag_annotation.annotation_map.keys()):
            if sub == marker:
                label_value = idx

        for coor in tissue_tag_annotation.positions.loc[list(combined_gene_indices), ["pxl_row", "pxl_col"]].to_numpy():
            labels[disk((coor[0], coor[1]), r)] = label_value + 1

    tissue_tag_annotation.label_image = labels

    return tissue_tag_annotation if copy else None


def background_labels_intensity(shape, imarray, r, intensity_threshold=230, space_every_spots=10, label=1):
    """
    Generate background labels based on intensity (bright pixels in brightfield images).

    Parameters
    ----------
    shape : tuple
        Shape of the training labels array.
    imarray : numpy.ndarray
        RGB image used to identify bright background areas.
    r : float
        Radius of the spots.
    intensity_threshold : int, optional
        Threshold above which pixels are considered background. Default is 230.
    space_every_spots : int, optional
        Spacing between background spots. Default is 10.
    label : int, optional
        Label value for background spots. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array containing the background labels.
    """

    # Convert RGBA to grayscale using only RGB channels
    if imarray.shape[-1] == 4:  # RGBA
        grayscale = np.dot(imarray[..., :3], [0.2989, 0.5870, 0.1140])  # Standard grayscale conversion
    elif imarray.shape[-1] == 3:  # RGB
        grayscale = np.dot(imarray, [0.2989, 0.5870, 0.1140])
    else:
        raise ValueError("Unexpected number of channels in imarray.")

    # Identify bright pixels in the grayscale image (background areas)
    background_mask = grayscale > intensity_threshold

    training_labels = np.zeros(shape, dtype=np.uint8)
    grid = square_grid(r, shape, space_every_spots).T

    print(imarray.shape)

    for coor in grid:
        y, x = int(coor[1]), int(coor[0])  # Ensure integer indices
        if y >= background_mask.shape[0] or x >= background_mask.shape[1]:  # Avoid out-of-bounds indexing
            continue
        if np.any(background_mask[y, x]):  # Use `.any()` if needed
            training_labels[disk((y, x), r, shape=shape)] = label

    return training_labels


def square_grid(spot_size, shape, space_every_spots):
    """
    Generate a square grid

    Parameters
    ----------
    spot_size : float
        Size of the spots.
    shape : tuple
        Shape of the grid (height, width).
    space_every_spots : int
        Spacing between background spots.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates of the grid.
    """
    # Define step sizes
    dx = spot_size * space_every_spots  # Horizontal spacing
    dy = spot_size * space_every_spots  # Vertical spacing

    # Generate meshgrid for a square grid
    x_coords = np.arange(spot_size, shape[0] - spot_size, dx)
    y_coords = np.arange(spot_size, shape[1] - spot_size, dy)

    gx, gy = np.meshgrid(x_coords, y_coords)

    # Stack the x and y coordinates
    positions = np.vstack([gy.ravel(), gx.ravel()])

    return positions


def median_filter(tissue_tag_annotation, filter_radius=10, downsampling_factor=1, copy=False):
    """
    Apply a median filter to the label image of the TissueTagAnnotation object.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image.
    filter_radius : int, optional
        Radius of the median filter. Default is 10.
    downsampling_factor: integer, optional
        Downsampling factor to resize label_image prior to running the median filter. Default value of 1 means no downsampling is performed.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the label_image in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with filtered label_image if copy is True,otherwise None.
    """

    from skimage.filters import median
    from skimage.morphology import disk

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation
    label_image_shape = tissue_tag_annotation.label_image.shape
    r = int(filter_radius * tissue_tag_annotation.ppm)

    if downsampling_factor > 1:
        r = r // downsampling_factor
        tissue_tag_annotation.label_image = downsample_segmentation(tissue_tag_annotation.label_image,
                                                                    (downsampling_factor, downsampling_factor))[0]

    tissue_tag_annotation.label_image = median(tissue_tag_annotation.label_image, footprint=disk(r))

    if downsampling_factor > 1:
        tissue_tag_annotation.label_image = cv2.resize(tissue_tag_annotation.label_image, label_image_shape[::-1],
                                          interpolation=cv2.INTER_NEAREST)

    return tissue_tag_annotation if copy else None


def assign_annotation_label_to_positions(tissue_tag_annotation, annotation_column='annotation', copy=False):
    """
    Assign annotation to each cell in positions data frame based on the cell centroid co-ordinates.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object with label_image and positions data frame.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.
    copy: bool, optional
        Return a copy of TissueTagAnnotation object rather than updating the positions data frame in the original object.

    Returns
    -------
    None | TissueTagAnnotation
        TissueTagAnnotation object with updated positions data frame if copy is True,otherwise None.
    """
    if tissue_tag_annotation.label_image is None:
        raise ValueError("Label image is missing. Please annotate the image first.")

    if tissue_tag_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")

    if tissue_tag_annotation.positions is None:
        raise ValueError("Positions data frame is missing. Please provide positions data frame.")

    tissue_tag_annotation = cp.deepcopy(tissue_tag_annotation) if copy else tissue_tag_annotation

    annotation_label_list = {i + 1: v for i, v in enumerate(tissue_tag_annotation.annotation_map.keys())}

    def get_annotation(row):
        annotation_id = tissue_tag_annotation.label_image[int(np.round(row["pxl_row"])), int(np.round(row["pxl_col"]))]
        return annotation_label_list.get(annotation_id, "Unknown")

    tissue_tag_annotation.positions[annotation_column] = tissue_tag_annotation.positions.apply(get_annotation, 1)

    return tissue_tag_annotation if copy else None


def plot_cell_label_annotations(tissue_tag_annotation, cell_diameter=5.0, annotation_column='annotation', alpha=0.8):
    """
    Helper function to plot the cell annotation labels on the image.

    Parameters
    ----------
    tissue_tag_annotation: TissueTagAnnotation
        TissueTagAnnotation object with image, label_image and annotation map.
    cell_diameter: float, optional
        Desired marker diameter in microns. Defaults to 5.0.
        Recommended values are 55 for Visium and 2, 8 or 16 depending on bin size selected for Visium HD.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.
    alpha : float, optional
        Blending factor, by default 0.8.

    Returns
    -------
    None
    """

    if tissue_tag_annotation.positions is None:
        raise ValueError("Positions data frame is missing. Please provide positions data frame.")
    if annotation_column not in tissue_tag_annotation.positions.columns is None:
        raise ValueError("Annotation column in posititions data frame is missing. Please assign annotation to cells first.")

    fig, ax = plt.subplots(figsize=(10, 10), dpi = 100)

    base_img = np.zeros(tissue_tag_annotation.image.shape, dtype=tissue_tag_annotation.image.dtype)
    base_img[:, :, :] = (alpha * tissue_tag_annotation.image[:, :, :])
    base_img[:, :, 3] = 255
    ax.imshow(base_img, origin='lower')

    marker_size_pixels = cell_diameter * tissue_tag_annotation.ppm
    for ann, color in tissue_tag_annotation.annotation_map.items():
        selected_position = tissue_tag_annotation.positions[tissue_tag_annotation.positions[annotation_column] == ann]
        zipped = np.broadcast(selected_position["pxl_col"], selected_position["pxl_row"], marker_size_pixels * 0.5)
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
        collection = PatchCollection(patches)
        collection.set_facecolor(color)
        collection.set_edgecolor(color)
        collection.set_alpha(1-alpha)
        ax.add_collection(collection)

    plt.show()
    plt.close(fig)