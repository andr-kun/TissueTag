import json
import pickle
from dataclasses import dataclass
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import h5py

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
        Saves the annotated image and related data to an HDF5 file.

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
    Loads the annotated image and related data from an HDF5 file.

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
        if to plot the loaded image. default is True.

    Returns
    -------
    numpy.ndarray
        The processed image.
    float
        The pixels per microns of the input image.
    float
        The pixels per microns of the output image.

    Raises
    ------
    ValueError
        If 'ppm_image' is not provided and cannot be extracted from the image metadata.
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
    res_in_ppm = None,
    fullres_path = None,
    header = None,
    plot = True,
    in_tissue = True,
) -> TissueTagAnnotation:
    """
    Reads 10X Visium image data from SpaceRanger (v1.3.0).

    Parameters
    ----------
    spaceranger_dir_path : str
        Path to the 10X SpaceRanger output folder.
    use_resolution : {'hires', 'lowres', 'fullres'}, optional
        Desired image resolution. 'fullres' refers to the original image that was sent to SpaceRanger
        along with sequencing data. If 'fullres' is specified, `fullres_path` must also be provided.
        Defaults to 'hires'.
    res_in_ppm : float, optional
        Used when working with full resolution images to resize the full image to a specified pixels per
        microns.
    fullres_path : str, optional
        Path to the full resolution image used for mapping. This must be specified if `use_resolution` is
        set to 'fullres'.
    header : int, optional (defa
        newer SpaceRanger could need this to be set as 0. Default is None.
    plot : Boolean
        if to plot the visium object to scale
    in_tissue : Boolean
        if to take only tissue spots or all visium spots

    Returns
    -------
    numpy.ndarray
        The processed image.
    float
        The pixels per microns of the image.
    pandas.DataFrame
        A DataFrame containing information on the tissue positions.

    Raises
    ------
    AssertionError
        If 'use_resolution' is set to 'fullres' but 'fullres_path' is not specified.
    """
    plt.figure(figsize=[12,12])

    spotsize = 55 #um spot size of a visium spot

    scalef = json.load(open(spaceranger_dir_path+'spatial/scalefactors_json.json','r'))
    if use_resolution=='fullres':
        assert fullres_path is not None, 'if use_resolution=="fullres" fullres_path has to be specified'

    df = pd.read_csv(spaceranger_dir_path+'spatial/tissue_positions_list.csv',header=header)
    if header==0:
        df = df.set_index(keys='barcode')
        if in_tissue:
            df = df[df['in_tissue']>0] # in tissue
         # turn df to mu
        fullres_ppm = scalef['spot_diameter_fullres']/spotsize
        df['pxl_row_in_fullres'] = df['pxl_row_in_fullres']/fullres_ppm
        df['pxl_col_in_fullres'] = df['pxl_col_in_fullres']/fullres_ppm
    else:
        df = df.set_index(keys=0)
        if in_tissue:
            df = df[df[1]>0] # in tissue
         # turn df to mu
        fullres_ppm = scalef['spot_diameter_fullres']/spotsize
        df['pxl_row_in_fullres'] = df[4]/fullres_ppm
        df['pxl_col_in_fullres'] = df[5]/fullres_ppm


    if use_resolution=='fullres':
        im = Image.open(fullres_path)
        ppm = fullres_ppm
    else:
        im = Image.open(spaceranger_dir_path+'spatial/tissue_'+use_resolution+'_image.png')
        ppm = scalef['spot_diameter_fullres']*scalef['tissue_'+use_resolution+'_scalef']/spotsize


    if res_in_ppm:
        width, height = im.size
        newsize = (int(width*res_in_ppm/ppm), int(height*res_in_ppm/ppm))
        im = im.resize(newsize,Image.Resampling.LANCZOS)
        ppm = res_in_ppm

    # translate from mu to pixel
    df['pxl_col'] = df['pxl_col_in_fullres']*ppm
    df['pxl_row'] = df['pxl_row_in_fullres']*ppm


    im = im.convert("RGBA")

    if plot:
        coordinates = np.vstack((df['pxl_col'],df['pxl_row']))
        plt.imshow(im,origin='lower')
        plt.plot(coordinates[0,:],coordinates[1,:],'.')
        plt.title( 'ppm - '+str(ppm))

    return TissueTagAnnotation(np.array(im), ppm, positions=df)


def simonson_vHE(dapi_image, eosin_image):
    """
    Create virtual H&E images using DAPI and eosin images.
    from the developer website:
    The method is applied to data on a multiplex/Keyence microscope to produce virtual H&E images
    using fluorescence data. If you find it useful, consider citing the relevant article:
    Creating virtual H&E images using samples imaged on a commercial multiplex platform
    Paul D. Simonson, Xiaobing Ren, Jonathan R. Fromm
    doi: https://doi.org/10.1101/2021.02.05.21249150

    Parameters
    ----------
    dapi_image : ndarray
        DAPI image data.
    eosin_image : ndarray
        Eosin image data.

    Returns
    -------
    ndarray
        Virtual H&E image.
    """

    def createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin):
        new_image = np.empty([dapi_image.shape[0], dapi_image.shape[1], 4])
        new_image[:,:,0] = background[0] + (1 - background[0]) * np.exp(- k1 * beta_DAPI[0] * dapi_image - k2 * beta_eosin[0] * eosin_image)
        new_image[:,:,1] = background[1] + (1 - background[1]) * np.exp(- k1 * beta_DAPI[1] * dapi_image - k2 * beta_eosin[1] * eosin_image)
        new_image[:,:,2] = background[2] + (1 - background[2]) * np.exp(- k1 * beta_DAPI[2] * dapi_image - k2 * beta_eosin[2] * eosin_image)
        new_image[:,:,3] = 1
        new_image = new_image*255
        return new_image.astype('uint8')

    k1 = k2 = 0.001

    background = [0.25, 0.25, 0.25]

    beta_DAPI = [9.147, 6.9215, 1.0]

    beta_eosin = [0.1, 15.8, 0.3]

    dapi_image = dapi_image[:,:,0]+dapi_image[:,:,1]
    eosin_image = eosin_image[:,:,0]+eosin_image[:,:,1]

    print(dapi_image.shape)
    return createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin)
