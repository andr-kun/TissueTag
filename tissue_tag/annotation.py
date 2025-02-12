import base64
import copy
import pickle
import random
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Optional
from collections import OrderedDict

import bokeh
import holoviews as hv
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import panel as pn
from PIL import Image, ImageDraw, ImageFont, ImageColor
from bokeh.models import FreehandDrawTool, PolyDrawTool
from holoviews.operation import datashader as hd
from matplotlib import pyplot as plt
from packaging import version
from skimage import feature, future
from skimage.draw import polygon, disk
from sklearn.ensemble import RandomForestClassifier

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
    Attaches a FreehandDrawTool and syncs the datasource.
    """

    def __init__(self, empty_value=None, drag=True, num_objects=0, show_vertices=False, vertex_style={}, styles={},
                 tooltip=None, icon_colour="black", **params):
        self.icon_colour = icon_colour
        super().__init__(empty_value, drag, num_objects, show_vertices, vertex_style, styles, tooltip, **params)


class CustomPolyDrawCallback(hv.plotting.bokeh.callbacks.GlyphDrawCallback):

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
        kwargs['icon'] = create_icon(stream.tooltip[0], stream.icon_colour)
        poly_tool = PolyDrawTool(
            drag=all(s.drag for s in self.streams), renderers=renderers,
            **kwargs
        )
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)


class SynchronisedFreehandDrawLink(hv.plotting.links.Link):
    """
    This custom class is a helper designed for creating synchronised FreehandDraw tools.
    """

    _requires_target = True


class SynchronisedFreehandDrawCallback(hv.plotting.bokeh.LinkCallback):
    """
    This custom class implements the method to synchronise data between two FreehandDraw tools by manually updating
    the data_source of the linked tools.
    """

    source_model = "cds"
    source_handles = ["plot"]
    target_model = "cds"
    target_handles = ["plot"]
    on_source_changes = ["data"]
    on_target_changes = ["data"]

    source_code = """
        target_cds.data = source_cds.data
        target_cds.change.emit()
    """

    target_code = """
        source_cds.data = target_cds.data
        source_cds.change.emit()
    """

# Overload the callback from holoviews to use the custom FreeHandDrawCallback class. Probably not safe.
hv.plotting.bokeh.callbacks.Stream._callbacks['bokeh'].update({
    CustomFreehandDraw: CustomFreehandDrawCallback,
    CustomPolyDraw: CustomPolyDrawCallback
})

# Register the callback class to the link class
SynchronisedFreehandDrawLink.register_callback('bokeh', SynchronisedFreehandDrawCallback)

@dataclass
class LabelAnnotation:
    label_image: Optional[np.array] = None
    annotation_map: Optional[dict] = None


def to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="png")
    data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{data}'


def create_icon(name, color):
    font_size = 28
    img = Image.new('RGBA', (30, 30), (255, 0, 0, 0))
    ImageDraw.Draw(img).text((5, 2), name, fill=tuple((np.array(matplotlib.colors.to_rgb(color)) * 255).astype(int)),
                             font=ImageFont.truetype(font_path, font_size), stroke_width=0.1, stroke_fill=(0, 0, 0, 255))
    if version.parse(bokeh.__version__) < version.parse("3.1.0"):
        img = to_base64(img)
    return img


# Annotation functions

def annotator(imarray, label_annotation, plot_size=1024, invert_y=False, use_datashader=False,
              unassigned_colour="yellow"):
    """
    Interactive annotation tool with line annotations using Panel tabs for toggling between morphology and annotation.

    Parameters
    ----------
    imarray: np.array
        Image in numpy array format.
    labels: np.array
        Label image in numpy array format.
    anno_dict: dict
        Dictionary of structures to annotate and colors for the structures.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y :boolean
        invert plot along y axis
    use_datashader : Boolean, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.

    Returns
    -------
    Panel Tabs object
        A Tabs object containing the annotation and image panels.
    dict
        Dictionary of Bokeh renderers for each annotation.
    """
    import logging
    logging.getLogger('bokeh.core.validation.check').setLevel(logging.ERROR)

    if label_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide annotation map.")
    else:
        label_annotation.annotation_map = OrderedDict(label_annotation.annotation_map)
        label_annotation.annotation_map["unassigned"] = unassigned_colour
        label_annotation.annotation_map.move_to_end("unassigned", last=False)

    if label_annotation.label_image is None:
        label_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.uint8)
        label_annotation.label_image = label_image
        annotation = rgb_from_labels(LabelAnnotation(label_image=label_image, annotation_map={'default': '#00000000'}))
    else:
        annotation = rgb_from_labels(label_annotation)

    annotation_c = annotation.astype('uint8').copy()
    if not invert_y:
        annotation_c = np.flip(annotation_c, 0)

    imarray_c = imarray.astype('uint8').copy()
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
    for key in label_annotation.annotation_map.keys():
        path_dict[key] = hv.Path([]).opts(color=label_annotation.annotation_map[key], line_width=5, line_alpha=0.7)
        render_dict[key] = CustomFreehandDraw(source=path_dict[key], num_objects=200, tooltip=key,
                                              icon_colour=label_annotation.annotation_map[key])

        plot_list.append(path_dict[key])

    tab_object = pn.panel(hd.Overlay(plot_list).collate())
    # Create the tabbed view
    p = pn.Column(pn.Row(label_opacity, update_button, revert_button), tab_object)

    previous_labels = label_annotation.label_image.copy()


    def update_annotator(event):
        nonlocal tab_object, previous_labels, revert_button

        if not event:
            return

        tab_object.loading = True
        update_button.disabled = True

        previous_labels = label_annotation.label_image.copy()
        updated_labels = label_annotation.label_image.copy()
        for idx, a in enumerate(render_dict.keys()):
            if render_dict[a].data['xs']:
                for o in range(len(render_dict[a].data['xs'])):
                    x = np.array(render_dict[a].data['xs'][o]).astype(int)
                    y = np.array(render_dict[a].data['ys'][o]).astype(int)
                    rr, cc = polygon(y, x)
                    inshape = np.where(
                        np.array(label_annotation.label_image.shape[0] > rr) & np.array(0 < rr) & np.array(label_annotation.label_image.shape[1] > cc) & np.array(
                            0 < cc))[0]
                    updated_labels[rr[inshape], cc[inshape]] = idx + 1

        label_annotation.label_image = updated_labels

        annotation = rgb_from_labels(label_annotation)
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

        label_annotation.label_image = previous_labels
        annotation = rgb_from_labels(label_annotation)
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


def rgb_from_labels(label_annotation):
    """
    Helper function to plot from label images.

    Parameters
    ----------
    labelimage: np.array
        Label image with pixel values corresponding to labels.
    colors: list
        Colors corresponding to pixel values for plotting.

    Returns
    -------
    np.array
        Annotation image.
    """
    labelimage_rgb = np.zeros((label_annotation.label_image.shape[0], label_annotation.label_image.shape[1], 4))

    colours = list(label_annotation.annotation_map.values())
    for c in range(len(colours)):
        color = ImageColor.getcolor(colours[c], "RGBA")
        labelimage_rgb[label_annotation.label_image == c + 1, 0:4] = np.array(color)

    return labelimage_rgb.astype('uint8')


def sk_rf_classifier(im, label_annotation, plot=True):
    """
    A simple random forest pixel classifier from sklearn.

    Parameters
    ----------
    im : array
        The actual image to predict the labels from, should be the same size as training_labels.
    training_labels : array
        Label image with pixel values corresponding to labels.
    anno_dict: dict
        Dictionary of structures to annotate and colors for the structures.
     plot : boolean, optional
        if to plot the loaded image. default is True.

    Returns
    -------
    array
        Predicted label map.
    """

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=~True,
                            sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1)

    features = features_func(im)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(label_annotation.label_image, features, clf)

    trained_labels = LabelAnnotation(annotation_map=label_annotation.annotation_map)
    trained_labels.label_image = future.predict_segmenter(features, clf)

    if plot:
        labels_rgb = rgb_from_labels(trained_labels)
        overlay_labels(im,labels_rgb,alpha=0.7)

    return trained_labels

def overlay_labels(im1, im2, alpha=0.8, show=True):
    """
    Helper function to merge 2 images.

    Parameters
    ----------
    im1 : array
        1st image.
    im2 : array
        2nd image.
    alpha : float, optional
        Blending factor, by default 0.8.
    show : bool, optional
        If to show the merged plot or not, by default True.

    Returns
    -------
    array
        The merged image.
    """
    # Ensure `im1` has 4 channels if im2 has 4 channels
    if im2.shape[-1] == 4:
        im1 = np.dstack([im1, np.full((*im1.shape[:2], 1), 255, dtype=im1.dtype)])  # Convert im1 to RGBA

    #generate overlay image
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape,dtype=im1.dtype)
    out_img[:,:,:] = (alpha * im1[:,:,:]) + ((1-alpha) * im2[:,:,:])
    out_img[:,:,3] = 255
    if show:
        plt.imshow(out_img,origin='lower')
    return out_img


def plot_labels(imarray, label_annotation, alpha=0.8, show=True):
    """
    Helper function to plot the labels on the image.

    Parameters
    ----------
    imarray : array
        Image to plot the labels on.
    labels : array
        Label image with pixel values corresponding to labels.
    alpha : float, optional
        Blending factor, by default 0.8.
    show : bool, optional
        If to show the merged plot or not, by default True.

    Returns
    -------
    array
        The merged image.
    """

    labels_rgb = rgb_from_labels(label_annotation)
    return overlay_labels(imarray, labels_rgb, alpha, show)

def save_annotation(folder, file_name, label_annotation, ppm):
    """
    Saves the annotated image as .tif and in addition saves the translation
    from annotations to labels in a pickle file.

    Parameters
    ----------
    folder : str
        Folder where to save the annotations.
    label_image : numpy.ndarray
        Labeled image.
    file_name : str
        Name for tif image and pickle.
    anno_names : list
        Names of annotated objects.
    anno_colors : list
        Colors of annotated objects.
    ppm : float
        Pixels per microns.
    """

    label_image = Image.fromarray(label_annotation.label_image)
    label_image.save(folder + file_name + '.tif')
    with open(folder + file_name + '.pickle', 'wb') as handle:
        pickle.dump(label_annotation.annotation_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + file_name + '_ppm.pickle', 'wb') as handle:
        pickle.dump({'ppm': ppm}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_annotation(folder, file_name):
    """
    Loads the annotated image from a .tif file and the translation from annotations
    to labels from a pickle file.

    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.

    Returns
    -------
    tuple
        Returns annotation image, annotation order, pixels per microns, and annotation color.
        If `load_colors` is False, annotation color is not returned.
    """

    imP = Image.open(folder + file_name + '.tif')

    ppm = imP.info['resolution'][0]
    im = np.array(imP)

    print(f'loaded annotation image - {file_name} size - {str(im.shape)}')
    with open(folder + file_name + '.pickle', 'rb') as handle:
        annotation_map = pickle.load(handle)
        print('loaded annotation map')
        print(annotation_map)
    with open(folder + file_name + '_ppm.pickle', 'rb') as handle:
        ppm = pickle.load(handle)
        print('loaded ppm')
        print(ppm)

    return LabelAnnotation(im, annotation_map), ppm['ppm']


def segmenter(imarray, label_annotation, plot_size=1024, use_datashader=False, invert_y=False,
              annotation_prefix="object"):
    """
    Interactive annotation tool with line annotations using Panel tabs for toggling between morphology and annotation.
    The principle is that selecting closed/semiclosed shaped that will later be filled according to the proper annotation.

    Parameters
    ----------
    imarray : numpy.ndarray
        Image in numpy array format.
    annotation : numpy.ndarray
        Label image in numpy array format.
    anno_dict : dict
        Dictionary of structures to annotate and colors for the structures.
    plot_size: int, default=1024
        Figure size for plotting.
    use_datashader : Boolean, optional
        If we should use datashader for rendering the image. Recommended for high resolution image. Default is False.

    Returns
    -------
    Panel Tabs object
        A Tabs object containing the annotation and image panels.
    dict
        Dictionary containing the Bokeh renderers for the annotation lines.
    """

    if label_annotation.label_image is None:
        label_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.uint8)
        label_annotation.label_image = label_image
        label_annotation.annotation_map = OrderedDict({})
        annotation = rgb_from_labels(LabelAnnotation(label_image=label_image,
                                                     annotation_map=label_annotation.annotation_map))
    else:
        annotation = rgb_from_labels(label_annotation)

    # convert label image to rgb for annotation
    annotation = rgb_from_labels(label_annotation)

    annotation_c = annotation.astype('uint8').copy()
    if not invert_y:
        annotation_c = np.flip(annotation_c, 0)

    imarray_c = imarray.astype('uint8').copy()
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

    previous_label = label_annotation.label_image
    previous_annotation_map = label_annotation.annotation_map

    def update_segmenter(event):
        nonlocal tab_object, previous_label, previous_annotation_map, revert_button

        if not event:
            return

        tab_object.loading = True
        update_button.disabled = True

        colorpool = ['green', 'cyan', 'brown', 'magenta', 'blue', 'red', 'orange']

        previous_label = label_annotation.label_image.copy()
        previous_annotation_map = label_annotation.annotation_map.copy()

        existing_object_count = len(label_annotation.annotation_map.keys()) + 1
        print(existing_object_count)
        if erase_object.data['xs']:
            print("Erasing")
            for o in range(len(erase_object.data['xs'])):
                x = np.array(erase_object.data['xs'][o]).astype(int)
                y = np.array(erase_object.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = (label_annotation.label_image.shape[0] > rr) & (0 < rr) & (label_annotation.label_image.shape[1] > cc) & (
                            0 < cc)  # make sure pixels outside the image are ignored
                label_annotation.label_image[rr[inshape], cc[inshape]] = 0

        if draw_object.data['xs']:
            print("Updating")
            for o in range(len(draw_object.data['xs'])):
                x = np.array(draw_object.data['xs'][o]).astype(int)
                y = np.array(draw_object.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = (label_annotation.label_image.shape[0] > rr) & (0 < rr) & (label_annotation.label_image.shape[1] > cc) & (
                            0 < cc)  # make sure pixels outside the image are ignored
                label_annotation.label_image[rr[inshape], cc[inshape]] = existing_object_count + o
                label_annotation.annotation_map[annotation_prefix + '_' + str(existing_object_count+o)] = (
                    random.choice(colorpool))

        annotation = rgb_from_labels(label_annotation)
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

        label_annotation.label_image = previous_label.copy()
        label_annotation.annotation_map = previous_annotation_map.copy()

        annotation = rgb_from_labels(label_annotation)
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

def gene_labels_from_adata(adata, df, gene_markers, imarray, label_annotation, r, every_x_spots=101, unassigned_colour="yellow"):
    """
    Assign labels to training spots based on gene expression from an existing AnnData object.

    Parameters
    ----------
    adata : AnnData
        Pre-loaded AnnData object containing gene expression data.
    df : pandas.DataFrame
        DataFrame containing spot coordinates.
    gene_markers : dict
        Dictionary mapping markers to genes.
    imarray : np.ndarray
        Image to overlay labels onto.
    label_annotation : LabelAnnotation
        Object storing label image and annotation map.
    r : float
        Radius of the spots.
    every_x_spots : int, optional
        Spacing between background labels. Higher value means fewer spots. Default is 101.
    unassigned_colour : str, optional
        Color for unassigned labels. Default is "yellow".

    Returns
    -------
    LabelAnnotation
        Updated LabelAnnotation object containing the training labels.
    """

    if label_annotation.label_image is not None:
        print("Label image is not empty. Will replace with an empty label_image.")

    if label_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide an annotation map.")
    else:
        label_annotation.annotation_map = OrderedDict(label_annotation.annotation_map)
        label_annotation.annotation_map["unassigned"] = unassigned_colour
        label_annotation.annotation_map.move_to_end("unassigned", last=False)

    # Initialize label image
    label_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.uint8)
    label_annotation.label_image = label_image

    # Filter adata to match df indices
    adata = adata[df.index.intersection(adata.obs.index)]

    # Extract coordinates
    coordinates = np.array(df.loc[:, ["pxl_col", "pxl_row"]])
    labels = background_labels(label_annotation.label_image.shape[:2], coordinates.T, every_x_spots=every_x_spots, r=r)

    # Assign labels based on gene expression
    for marker, (gene, top_n) in gene_markers.items():
        print(f"Processing marker: {marker}, Gene: {gene}")

        GeneIndex = np.where(adata.var_names.str.fullmatch(gene))[0]
        if GeneIndex.size == 0:
            print(f"Warning: Gene {gene} not found in AnnData. Skipping.")
            continue

        # Normalize gene expression
        from scanpy.pp import normalize_total
        normalize_total(adata)

        # Extract gene expression data
        GeneData = adata.X[:, GeneIndex].todense()
        SortedExp = np.argsort(GeneData, axis=0)[::-1]  # Sort in descending order

        # Select top N spots based on gene expression
        list_gene = adata.obs.index[np.array(np.squeeze(SortedExp[:top_n]))[0]]

        # Assign labels
        for idx, sub in enumerate(label_annotation.annotation_map.keys()):
            if sub == marker:
                label_value = idx

        for coor in df.loc[list_gene, ["pxl_row", "pxl_col"]].to_numpy():
            labels[disk((coor[0], coor[1]), r)] = label_value + 1

    label_annotation.label_image = labels

    return label_annotation

def gene_labels(path, df, gene_markers, imarray, label_annotation, r, every_x_spots = 101, unassigned_colour="yellow"):
    """
    Assign labels to training spots based on gene expression.

    Parameters
    ----------
    path : string
        path to visium object (cellranger output folder)
    df : pandas.DataFrame
        DataFrame containing spot coordinates.
    labels : numpy.ndarray
        Array for storing the training labels.
    gene_markers : dict
        Dictionary mapping markers to genes.
    annodict : dict
        Dictionary mapping markers to annotation names.
    r : float
        Radius of the spots.
    every_x_spots : integer
        spacing of background labels to generate, higher number means less spots. Default is 100

    Returns
    -------
    LabelAnnotation
        LabelAnnotation object containing the training label.
    """

    if label_annotation.label_image is not None:
        raise ValueError("Label image is not empty. Please provide a label_annotation object with empty label_image.")

    if label_annotation.annotation_map is None:
        raise ValueError("Annotation map is missing. Please provide annotation map.")
    else:
        label_annotation.annotation_map = OrderedDict(label_annotation.annotation_map)
        label_annotation.annotation_map["unassigned"] = unassigned_colour
        label_annotation.annotation_map.move_to_end("unassigned", last=False)

    label_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.uint8)
    label_annotation.label_image = label_image

    import scanpy
    adata = scanpy.read_visium(path,count_file='raw_feature_bc_matrix.h5')
    adata = adata[df.index.intersection(adata.obs.index)]
    coordinates = np.array(df.loc[:,['pxl_col','pxl_row']])
    labels = background_labels(label_annotation.label_image.shape[:2], coordinates.T, every_x_spots=every_x_spots, r=r)

    for m in list(gene_markers.keys()):
        print(gene_markers[m])
        GeneIndex = np.where(adata.var_names.str.fullmatch(gene_markers[m][0]))[0]
        scanpy.pp.normalize_total(adata)
        GeneData = adata.X[:, GeneIndex].todense()
        SortedExp = np.argsort(GeneData, axis=0)[::-1]
        list_gene = adata.obs.index[np.array(np.squeeze(SortedExp[range(gene_markers[m][1])]))[0]]
        for idx, sub in enumerate(list(label_annotation.annotation_map.keys())):
            if sub == m:
                back = idx
        for coor in df.loc[list_gene, ['pxl_row','pxl_col']].to_numpy():
            labels[disk((coor[0], coor[1]), r)] = back + 1

    label_annotation.label_image = labels

    return label_annotation


def background_labels(shape, coordinates, r, every_x_spots=10, label=1):
    """
    Generate background labels.

    Parameters
    ----------
    shape : tuple
        Shape of the training labels array.
    coordinates : numpy.ndarray
        Array containing the coordinates of the spots.
    r : float
        Radius of the spots.
    every_x_spots : int, optional
        Spacing between background spots. Default is 10.
    label : int, optional
        Label value for background spots. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array containing the background labels.
    """

    training_labels = np.zeros(shape, dtype=np.uint8)
    Xmin = np.min(coordinates[:, 0])
    Xmax = np.max(coordinates[:, 0])
    Ymin = np.min(coordinates[:, 1])
    Ymax = np.max(coordinates[:, 1])
    grid = hexagonal_grid(r, shape)
    grid = grid.T
    grid = grid[::every_x_spots, :]

    for coor in grid:
        training_labels[disk((coor[1], coor[0]), r,shape=shape)] = label


    for coor in coordinates.T:
        training_labels[disk((coor[1], coor[0]), r * 4,shape=shape)] = 0

    return training_labels


def hexagonal_grid(SpotSize, shape):
    """
    Generate a hexagonal grid.

    Parameters
    ----------
    SpotSize : float
        Size of the spots.
    shape : tuple
        Shape of the grid.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates of the grid.
    """

    helper = SpotSize
    X1 = np.linspace(helper, shape[0] - helper, round(shape[0] / helper))
    Y1 = np.linspace(helper, shape[1] - 2 * helper, round(shape[1] / (2 * helper)))
    X2 = X1 + SpotSize / 2
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1, Y1)
    Gx2, Gy2 = np.meshgrid(X2, Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1, positions2])
    return positions
