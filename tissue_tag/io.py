import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skimage.exposure
from PIL import Image, ImageEnhance, ImageOps, ImageColor
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import h5py
import gzip
import tifffile

@dataclass
class TissueTagAnnotation:
    image: np.array
    ppm: float
    label_image: Optional[np.array] = None
    annotation_map: Optional[dict] = None
    positions: Optional[pd.DataFrame] = None
    grid: Optional[pd.DataFrame] = None

    def save_annotation(self, file_path):
        """
        Saves the TissueTagAnnotation object into HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.
        """
        with h5py.File(file_path, 'w') as f:
            if self.image is not None:
                f.create_dataset('image', data=self.image)
            if self.ppm is not None:
                f.create_dataset('ppm', data=self.ppm)
            if self.label_image is not None:
                f.create_dataset('label_image', data=self.label_image)
            if self.annotation_map is not None:
                f.create_dataset('annotation_map', data=json.dumps(self.annotation_map))
        if self.positions is not None:
            self.positions.to_hdf(file_path, key="positions", mode="a")
        if self.grid is not None:
            self.grid.to_hdf(file_path, key="positions", mode="a")


def load_annotation(file_path):
    """
    Loads the TissueTagAnnotation object from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    TissueTagAnnotation
        The loaded TissueTagAnnotation object.
    """
    with h5py.File(file_path, 'r') as f:
        image = f['image'][:] if 'image' in f else None
        ppm = f['ppm'][()] if 'ppm' in f else None
        label_image = f['label_image'][:] if 'label_image' in f else None
        annotation_map = json.loads(f['annotation_map'][()]) if 'annotation_map' in f else None
        positions = None
        if 'positions' in f:
            positions = pd.read_hdf(file_path, key="positions")
        grid = None
        if 'grid' in f:
            grid = pd.read_hdf(file_path, key="grid")

    if image is not None:
        print(f'> loaded image - size - {str(image.shape)}')
    if ppm is not None:
        print(f'> loaded ppm: {ppm}')
    if label_image is not None:
        print(f'> loaded label image - size - {str(label_image.shape)}')
    if annotation_map is not None:
        print(f'> loaded annotation map:')
        print(annotation_map)
    if positions is not None:
        print('> loaded positions')
    if grid is not None:
        print('> loaded grid')
    return TissueTagAnnotation(image, ppm, label_image, annotation_map, positions, grid)


def read_image(
    path,
    ppm_image=None,
    ppm_out=1,
    contrast_factor=1,
    background_image_path=None,
    plot=True,
) -> TissueTagAnnotation:
    """
    Reads an H&E or fluorescent image and returns the image with optional enhancements.

    Parameters
    ----------
    path : str
        Path to the image. The image must be in a format supported by Pillow. Refer to
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for the list
        of supported formats.
    ppm_image : float, optional
        Pixels per microns of the input image. If not provided, this will be extracted from the image
        metadata with info['resolution']. If the metadata is not present, an error will be thrown.
    ppm_out : float, optional
        Pixels per microns of the output image. Defaults to 1.
    contrast_factor : int, optional
        Factor to adjust contrast for output image, typically between 2-5. Defaults to 1.
    background_image_path : str, optional
        Path to a background image. If provided, this image and the input image are combined
        to create a virtual H&E (vH&E). If not provided, vH&E will not be performed.
    plot : boolean, optional
        if to plot the loaded image. Defaults to True.

    Returns
    -------
    TissueTagAnnotation
        TissueTagAnnotation object containing the H&E or fluorescent image
    """

    im = Image.open(path)
    if not(ppm_image):
        try:
            ppm_image = im.info['resolution'][0]
            print('found ppm in image metadata!, its - '+str(ppm_image))
        except:
            print('could not find ppm in image metadata, please provide ppm value')
    width, height = im.size
    newsize = (int(width/ppm_image*ppm_out), int(height/ppm_image*ppm_out))
    # resize
    im = im.resize(newsize,Image.Resampling.LANCZOS)
    im = im.convert("RGBA")
    #increase contrast
    enhancer = ImageEnhance.Contrast(im)
    factor = contrast_factor
    im = enhancer.enhance(factor*factor)

    if background_image_path:
        im2 = Image.open(background_image_path)
        # resize
        im2 = im2.resize(newsize,Image.Resampling.LANCZOS)
        im2 = im2.convert("RGBA")
        #increase contrast
        enhancer = ImageEnhance.Contrast(im2)
        factor = contrast_factor
        im2 = enhancer.enhance(factor*factor)
        # virtual H&E
        # im2 = im2.convert("RGBA")
        im = simonson_vHE(np.array(im).astype('uint8'),np.array(im2).astype('uint8'))

    if plot:
        plt.figure(dpi=100)
        plt.imshow(im,origin='lower')
        plt.show()

    return TissueTagAnnotation(np.array(im), ppm_out)


def read_visium(
    spaceranger_dir_path,
    use_resolution='hires',
    ppm_out = None,
    mapped_image_path = None,
    in_tissue = True,
    plot = False,
) -> TissueTagAnnotation:
    """
    Reads 10X Visium data from SpaceRanger.

    Parameters
    ----------
    spaceranger_dir_path : str
        Directory containing Visium library data.
    use_resolution : {'hires', 'lowres', 'mapped_res'}, optional
        Desired image resolution. 'mapped_res' refers to the original image that was sent to SpaceRanger
        along with sequencing data. If 'mapped_res' is specified, `mapped_image_path` must also be provided.
        Defaults to 'hires'.
    ppm_out : float, optional
        Used when working with full resolution images ('mapped_res') to resize the full image to a specified pixels per
        microns.
    mapped_image_path : str, optional
        Path to the full resolution image used as input for SpaceRanger. This must be specified if `use_resolution` is
        set to 'mapped_res'.
    in_tissue : bool, optional
        Whether to include only tissue bins (default: True).
    plot : bool, optional
        Whether to plot the output image (default: False).


    Returns
    -------
    TissueTagAnnotation
        TissueTagAnnotation object containing the visium image and positions of the spots.
    """


    # Load scale factors
    scalefactors_file = spaceranger_dir_path + 'spatial/scalefactors_json.json'
    with open(scalefactors_file, "r") as f:
        scalefactors = json.load(f)

    fullres_ppm = scalefactors['spot_diameter_fullres'] / 55 # Harcoded as visium has a fixed spot size of 55um

    # Load tissue positions future proofing
    spaceranger_spatial_dir_path = Path(spaceranger_dir_path  + '/spatial')
    tissue_positions_file = next(
        (f for f in [
            spaceranger_spatial_dir_path / "tissue_positions.csv",
            spaceranger_spatial_dir_path / "tissue_positions_list.csv"
        ] if f.exists()),
        None
    )
    if tissue_positions_file.name.endswith("_list.csv"): # Output from space ranger < 2.0
        header = None
        col_names = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    else:
        header = 0
        col_names = None
    df = pd.read_csv(tissue_positions_file, header=header, names=col_names).set_index('barcode')

    if in_tissue:
        df = df[df['in_tissue'] > 0]
    # adjust to fullres
    df["pxl_row_in_fullres"] /= fullres_ppm
    df["pxl_col_in_fullres"] /= fullres_ppm

    # Load images
    spaceranger_spatial_path = Path(spaceranger_dir_path + '/spatial')

    # image file paths
    image_files = {
        "mapped_res": mapped_image_path,
        "hires": spaceranger_spatial_path / "tissue_hires_image.png",
        "lowres": spaceranger_spatial_path / "tissue_lowres_image.png",
    }

    if use_resolution == "mapped_res" and mapped_image_path is None:
        raise ValueError("Full resolution image path must be provided.")
    if use_resolution == "mapped_res":
        print('!!! Make sure this mapped_res image is the same one you used as spaceranger input !!!')

    im = Image.open(image_files[use_resolution])
    ppm_anno = fullres_ppm * scalefactors[f"tissue_{use_resolution}_scalef"] if use_resolution != "mapped_res" else fullres_ppm # adjust resolution to the image

    # rescale image to target
    if ppm_out:
        width, height = im.size
        new_size = (int(width * ppm_out / ppm_anno), int(height * ppm_out / ppm_anno))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        ppm_anno = ppm_out

    # Convert coordinates by the same scaling
    df["pxl_col"] = df["pxl_col_in_fullres"] * ppm_anno
    df["pxl_row"] = df["pxl_row_in_fullres"] * ppm_anno

    # Convert image to array
    im = im.convert("RGBA")
    im = np.array(im)

    if plot:
        plot_10x_spatial_image(im, df, ppm_anno, 55, dpi=300, blowup_size_um=250, technology="Visium", image_info="Image resolution: " + use_resolution)

    return TissueTagAnnotation(image=im,ppm=ppm_anno,positions=df)

def read_visium_hd(
    spaceranger_dir_path,
    bin_resolution  = '16',
    use_resolution = "hires",
    ppm_out = None,
    mapped_image_path = None,
    in_tissue = True,
    plot = False,
):
    """
    Reads 10X Visium HD data from SpaceRanger, including spatial image and metadata.

    Parameters
    ----------
    spaceranger_dir_path : str
        Directory containing Visium HD library data.
    bin_resolution : str, optional
        Resolution of the Visium HD bins to use. Binning level can be - '02','08','16'. Defaults to 16.
    use_resolution : {'hires', 'lowres', 'mapped_res'}, optional
        Desired image resolution. 'mapped_res' refers to the original image that was sent to SpaceRanger
        along with sequencing data. If 'mapped_res' is specified, `mapped_image_path` must also be provided.
        Defaults to 'hires'.
    ppm_out : float, optional
        Target resolution in pixels per micron.
    mapped_image_path : str, optional
        Path to the full resolution image used as input for SpaceRanger. This must be specified if `use_resolution` is
        set to 'mapped_res'.
    in_tissue : bool, optional
        Whether to include only tissue bins. Defaults to True.
    plot : bool, optional
        Whether to plot the output image. Defaults to False.


    Returns
    -------
    TissueTagAnnotation
        TissueTagAnnotation object containing the visium HD image and positions of the spots.
    """


    # Load scale factors
    scalefactors_file = spaceranger_dir_path + f'/binned_outputs/square_0{bin_resolution}um/spatial/scalefactors_json.json'
    with open(scalefactors_file, "r") as f:
        scalefactors = json.load(f)

    fullres_ppm = 1/scalefactors["microns_per_pixel"] # get to micron scale from pixels

    # Load tissue positions future proofing
    spaceranger_spatial_dir_path = Path(spaceranger_dir_path + f'/binned_outputs/square_0{bin_resolution}um/spatial/')  # Convert to Path
    tissue_positions_file = next(
        (f for f in [
            spaceranger_spatial_dir_path / "tissue_positions.parquet",
            spaceranger_spatial_dir_path / "tissue_positions.csv",
            spaceranger_spatial_dir_path / "tissue_positions_list.csv"
        ] if f.exists()),
        None
    )
    if tissue_positions_file.suffix == ".csv":
        df = pd.read_csv(tissue_positions_file, index_col=0)
    elif tissue_positions_file.suffix == ".parquet":
        df = pd.read_parquet(tissue_positions_file).set_index("barcode")

    if in_tissue:
        df = df[df["in_tissue"] > 0]
    # adjust to fullres
    df["pxl_row_in_fullres"] /= fullres_ppm
    df["pxl_col_in_fullres"] /= fullres_ppm

    # Load images
    spaceranger_spatial_path = Path(spaceranger_dir_path + f'/spatial')

    # image file paths
    image_files = {
    "mapped_res": mapped_image_path,
    "hires": spaceranger_spatial_path / "tissue_hires_image.png",
    "lowres": spaceranger_spatial_path / "tissue_lowres_image.png",
    }

    if use_resolution == "mapped_res" and mapped_image_path is None:
        raise ValueError("Full resolution image path must be provided.")
    if use_resolution == "mapped_res":
        print('!!! Make sure this mapped_res image is the same one you used as spaceranger input !!!')

    im = Image.open(image_files[use_resolution])
    ppm_anno = fullres_ppm * scalefactors[f"tissue_{use_resolution}_scalef"] if use_resolution != "mapped_res" else fullres_ppm # adjust resolution to the image

    # rescale image to target
    if ppm_out:
        width, height = im.size
        new_size = (int(width * ppm_out / ppm_anno), int(height * ppm_out / ppm_anno))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        ppm_anno = ppm_out

    # Convert coordinates by the same scaling
    df["pxl_col"] = df["pxl_col_in_fullres"] * ppm_anno
    df["pxl_row"] = df["pxl_row_in_fullres"] * ppm_anno

    # Convert image to array
    im = im.convert("RGBA")
    im = np.array(im)

    # Call the plotting function if plot=True
    if plot:
        plot_10x_spatial_image(im, df, ppm_anno, int(bin_resolution), dpi=300, blowup_size_um=320, technology="VisiumHD", image_info="Image resolution: " + use_resolution)

    return TissueTagAnnotation(image=im,ppm=ppm_anno,positions=df)


def read_xenium(
    xeniumranger_dir_path,
    ppm_out = 1,
    image_quantiles = (0.05, 0.999),
    image_output = "fluorescence",
    fluorescence_channels = [0, 1, 2, 3],
    channel_colours = ["#0F73E6", "#F300A5", "#A4A400", "#008A00"],
    plot = False,
):
    """
    Reads 10X Xenium data from XeniumRanger, including spatial image and metadata.

    Parameters
    ----------
    xeniumranger_dir_path : str
        Directory containing Xenium library data.
    ppm_out : float, optional
        Target resolution in pixels per micron.
    image_quantiles : tuple, optional
        Quantiles to use for auto-scaling of fluorescence levels.
    image_output : str, optional
        Whether to generate virtual fluorescence image or a virtual H&E image as output.
        If fluorescence is selected, the channels loaded and the colour of the channels will be based on the
        fluorescence_channels and channel_colours parameters.
        If virtualHE is selected, the first channel will be used as the dapi image and
        the remaining channels will be stacked ans used as the eosin image.
    fluorescence_channels : list, optional
        List of channels to include in the virtual fluorescence image.
    channel_colours: str or list, optional
        Colours to use for the fluorescence channels. Can be set to "grayscale" or a list of 4 channel colours in hex format.
    plot : bool, optional
        Whether to plot the output image. Defaults to False.


    Returns
    -------
    TissueTagAnnotation
        TissueTagAnnotation object containing the Xenium morphology focus image and positions of the spots.
    """

    if image_output not in ["fluorescence", "virtualHE"]:
        raise ValueError("Image output must be 'fluorescence' or 'virtualHE'.")
    if fluorescence_channels is None or len(fluorescence_channels) == 0:
        raise ValueError("Need to specify at least one channel to load.")
    if channel_colours is None:
        raise ValueError("Need to specify channel colours as either 'grayscale' or a list of 4 channel colours in hex.")
    else:
        if not ((type(channel_colours) == str and channel_colours == "grayscale") or
                (type(channel_colours) == list and len(channel_colours) == 4)):
            raise ValueError(
                "Need to specify channel colours as either 'grayscale' or a list of 4 channel colours in hex.")

    # Load xenium metadata
    metadata_file = xeniumranger_dir_path + f'/experiment.xenium'
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    fullres_ppm = 1/metadata["pixel_size"] # get to micron scale from pixels

    # Load cell positions
    cell_position =  xeniumranger_dir_path + f'/cells.csv.gz'
    with gzip.open(cell_position, 'rt') as f:
        df = pd.read_csv(f, index_col=0)

    # Load images
    morphology_path = Path(xeniumranger_dir_path + f'/morphology_focus/')

    # image file paths
    image_files = sorted([f for f in morphology_path.glob("*.ome.tif") if f.is_file()])

    # calculate ppm for each pyramidal layer
    im_meta = tifffile.TiffFile(image_files[0]).pages[0]
    pyramidal_ppm = np.zeros(len(im_meta.pages)+1)
    pyramidal_ppm[0] = im_meta.shape[1]
    for i in range(len(im_meta.pages)):
        pyramidal_ppm[i+1] = im_meta.pages[i].shape[1]
    pyramidal_ppm /= pyramidal_ppm[0]
    pyramidal_ppm *= fullres_ppm

    # select pyramidal layer to load
    if ppm_out is None:
        ppm_out = fullres_ppm

    pyramid_layer = np.abs(pyramidal_ppm - ppm_out).argmin()
    pyramid_ppm = pyramidal_ppm[pyramid_layer]

    if image_output == "virtualHE":
        fluorescence_channels = [1, 2, 3, 0]
        channel_colours = "grayscale"

    stacked_im = None
    for channel in fluorescence_channels:
        im = tifffile.imread(image_files[channel], is_ome=False, level=pyramid_layer)

        low_quantile, high_quantile = np.quantile(im, q=image_quantiles[0]), np.quantile(im, q=image_quantiles[1])
        im[im > high_quantile] = high_quantile
        im = ((im - low_quantile) / (high_quantile - low_quantile) * 255).astype(np.uint8)
        im = np.flipud(im)

        im = Image.fromarray(im).convert("L")

        if ppm_out != pyramid_ppm:
            width, height = im.size
            new_size = (int(width / pyramid_ppm * ppm_out), int(height / pyramid_ppm * ppm_out))

            im = im.resize(new_size, Image.Resampling.LANCZOS)

        if channel_colours != "grayscale":
            im = ImageOps.colorize(im, black="black", white=ImageColor.getcolor(channel_colours[channel], "RGB"))
        im = im.convert("RGBA")

        if stacked_im is None:
            stacked_im = np.array(im)
        else:
            if image_output == "virtualHE" and channel == 0:
                stacked_im = simonson_vHE(np.array(im), stacked_im)
            else:
                proj_fun = np.max
                stacked_im = proj_fun(np.stack([stacked_im, np.array(im)], axis = 3), axis = 3)

        del(im)

    print(stacked_im.shape)

    # Convert coordinates by the same scaling
    df["pxl_col"] = df["x_centroid"] * ppm_out
    df["pxl_row"] = (stacked_im.shape[0] - 1) - df["y_centroid"] * ppm_out

    # Call the plotting function if plot=True
    if plot:
        plot_10x_spatial_image(stacked_im, df, ppm_out, 0.5, dpi=300, blowup_size_um=320, technology="Xenium", image_info=f"Output type: {image_output}", blowup_marker_multiplier=6)

    return TissueTagAnnotation(image=stacked_im,ppm=ppm_out,positions=df)


def simonson_vHE(dapi_image, eosin_image):
    """
    Create virtual H&E images using DAPI and eosin images.

    This function is based on the method described in:
    Creating virtual H&E images using samples imaged on a commercial multiplex platform
    Paul D. Simonson, Xiaobing Ren, Jonathan R. Fromm
    doi: https://doi.org/10.1101/2021.02.05.21249150

    Parameters
    ----------
    dapi_image : numpy.ndarray
        DAPI image data.
    eosin_image : numpy.ndarray
        Eosin image data.

    Returns
    -------
    numpy.ndarray
        Virtual H&E image.
    """

    k1 = k2 = 0.001

    background = [0.25, 0.25, 0.25]

    beta_DAPI = [9.147, 6.9215, 1.0]

    beta_eosin = [0.1, 15.8, 0.3]

    dapi_image = dapi_image.astype('float64')
    eosin_image = eosin_image.astype('float64')

    dapi_image = dapi_image[:,:,0]+dapi_image[:,:,1]
    eosin_image = eosin_image[:,:,0]+eosin_image[:,:,1]

    print(dapi_image.shape)

    # create the virtual H&E image
    new_image = np.empty([dapi_image.shape[0], dapi_image.shape[1], 4])
    new_image[:, :, 0] = background[0] + (1 - background[0]) * np.exp(
        - k1 * beta_DAPI[0] * dapi_image - k2 * beta_eosin[0] * eosin_image)
    new_image[:, :, 1] = background[1] + (1 - background[1]) * np.exp(
        - k1 * beta_DAPI[1] * dapi_image - k2 * beta_eosin[1] * eosin_image)
    new_image[:, :, 2] = background[2] + (1 - background[2]) * np.exp(
        - k1 * beta_DAPI[2] * dapi_image - k2 * beta_eosin[2] * eosin_image)
    new_image[:, :, 3] = 1
    new_image = new_image * 255

    return new_image.astype('uint8')


def plot_10x_spatial_image(
    image: np.ndarray,
    positions: pd.DataFrame,
    ppm: float,
    target_diameter_um: float = 16.0,
    dpi: int = 100,
    blowup_size_um: float = 500.0,
    technology: str = "Visium",
    image_info: str = None,
    blowup_marker_multiplier: float = 0.5
):
    """
    Plots 10x-based spatial data with a blowup region.

    Parameters
    ----------
        image: numpy.ndarray
            Visium image as a NumPy array.
        positions: pandas.DataFrame
            Pandas DataFrame containing spatial coordinates ('pxl_col', 'pxl_row').
        ppm: float
            Image resolution in pixels per micron.
        target_diameter_um: float, optional
            Desired marker diameter in microns. Defaults to 16.0.
        dpi: int, optional
            Figure DPI. Defaults to 100.
        blowup_size_um: float, optional
            Size of the blowup region in microns. Defaults to 500.0.
        image_info: str, optional
            Resolution of the image. Defaults to None.
        blowup_marker_multiplier: float, optional
            Marker size multiplier of the blowup region. Defaults to 0.5.

    Returns
    -------
        None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=dpi)

    # --- Main Plot (ax1) ---
    ax1.imshow(image, origin="lower")

    marker_size_pixels = target_diameter_um * ppm

    # Adapting scanpy implementation to plot the points
    zipped = np.broadcast(positions["pxl_col"], positions["pxl_row"], marker_size_pixels * 0.5)
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    collection = PatchCollection(patches)
    collection.set_facecolor("green")
    collection.set_edgecolor("green")
    collection.set_alpha(1)
    ax1.add_collection(collection)

    ax1.set_title(f"Visium Spatial Data (PPM: {ppm:.2f}, Bin size: {target_diameter_um}um{", " + image_info if image_info else ""})")

    # --- Blowup Region (ax2) ---

    # 1. Calculate Center of Data:
    center_x = positions["pxl_col"].mean()
    center_y = positions["pxl_row"].mean()

    # 2. Calculate Blowup Region Boundaries (in pixels):
    blowup_half_size_pixels = (blowup_size_um / 2) * ppm
    x_min = int(center_x - blowup_half_size_pixels)
    x_max = int(center_x + blowup_half_size_pixels)
    y_min = int(center_y - blowup_half_size_pixels)
    y_max = int(center_y + blowup_half_size_pixels)

    # 3.  Handle image boundaries:
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)  # image.shape[1] is width
    y_max = min(image.shape[0], y_max)  # image.shape[0] is height

    # 4. Extract the Blowup Region from the Image:
    blowup_im = image[y_min:y_max, x_min:x_max]

    # 5. Plot the Blowup Region:
    ax2.imshow(blowup_im, origin="lower")
    ax2.set_title(f"Blowup ({blowup_size_um}um x {blowup_size_um}um)")

    # 6.  Plot points WITHIN the blowup region on the blowup plot:
    df_blowup = positions[
        (positions["pxl_col"] >= x_min)
        & (positions["pxl_col"] < x_max)
        & (positions["pxl_row"] >= y_min)
        & (positions["pxl_row"] < y_max)
    ]

    df_blowup_adj = df_blowup.copy()
    df_blowup_adj["pxl_col"] -= x_min
    df_blowup_adj["pxl_row"] -= y_min

    marker_size_pixels = (target_diameter_um * ppm)

    zipped_blowup = np.broadcast(df_blowup_adj["pxl_col"], df_blowup_adj["pxl_row"], marker_size_pixels * blowup_marker_multiplier)
    patches_blowup = [Circle((x_, y_), s_) for x_, y_, s_ in zipped_blowup]
    collection_blowup = PatchCollection(patches_blowup)
    collection_blowup.set_facecolor("green")
    collection_blowup.set_edgecolor("green")
    collection_blowup.set_alpha(0.25)
    ax2.add_collection(collection_blowup)

    # Draw a rectangle on the main plot (ax1)
    rect = matplotlib.patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax1.add_patch(rect)

    # Remove x and y ticks from blowup
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()