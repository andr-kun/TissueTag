import json
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import skimage
import os
from PIL import Image,ImageEnhance
from scipy import interpolate
from skimage.draw import polygon
import skimage.transform
import skimage.draw
import scipy.ndimage

from tissue_tag.annotation import load_annotation

try:
    import scanpy as scread_visium
except ImportError:
    print('scanpy is not available')

Image.MAX_IMAGE_PIXELS = None


def read_image(
    path,
    ppm_image=None,
    ppm_out=1,
    contrast_factor=1,
    background_image_path=None,
    plot=True,
):
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
        
    return np.array(im),ppm_image,ppm_out

def read_visium(
    spaceranger_dir_path,
    use_resolution='hires',
    res_in_ppm = None,
    fullres_path = None,
    header = None,
    plot = True,
    in_tissue = True,
):
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
    
    return np.array(im), ppm, df


def complete_pixel_gaps(x,y):
    """
    Function to complete pixel gaps in a given x, y coordinates
    
    Parameters:
    x : list
        list of x coordinates
    y : list
        list of y coordinates
    
    Returns:
    new_x, new_y : tuple
        tuple of completed x and y coordinates
    """
    
    new_x_1 = []
    new_x_2 = []
    # iterate over x coordinate values
    for idx, px in enumerate(x[:-1]):
        # interpolate between each pair of x points
        interpolation = interpolate.interp1d(x[idx:idx+2], y[idx:idx+2])
        interpolated_x_1 = np.linspace(x[idx], x[idx+1], num=np.abs(x[idx+1] - x[idx] + 1))
        interpolated_x_2 = interpolation(interpolated_x_1).astype(int)
        # add interpolated values to new x lists
        new_x_1 += list(interpolated_x_1)
        new_x_2 += list(interpolated_x_2)

    new_y_1 = []
    new_y_2 = []
    # iterate over y coordinate values
    for idx, py in enumerate(y[:-1]):
        # interpolate between each pair of y points
        interpolation = interpolate.interp1d(y[idx:idx+2], x[idx:idx+2])
        interpolated_y_1 = np.linspace(y[idx], y[idx+1], num=np.abs(y[idx+1] - y[idx] + 1))
        interpolated_y_2 = interpolation(interpolated_y_1).astype(int)
        # add interpolated values to new y lists
        new_y_1 += list(interpolated_y_1)
        new_y_2 += list(interpolated_y_2)
    
    # combine x and y lists
    new_x = new_x_1 + new_y_2
    new_y = new_x_2 + new_y_1

    return new_x, new_y


def rescale_image(label_image, target_size):
    """
    Rescales label image to original image size.
        
    Parameters
    ----------
    label_image : numpy.ndarray
        Labeled image.
    target_size : tuple
        Final dimensions.
        
    Returns
    -------
    numpy.ndarray
        Rescaled image.
    """
    imP = Image.fromarray(label_image)
    newsize = (target_size[0], target_size[1])
    return np.array(imP.resize(newsize))


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


def generate_hires_grid(im, spot_to_spot, pixels_per_micron):
    """
    Creates a hexagonal grid of a specified size and density.
    
    Parameters
    ----------
    im : array-like
        Image to fit the grid on (mostly for dimensions).
    spot_to_spot : float
        determines the grid density.
    pixels_per_micron : float
        The resolution of the image in pixels per micron.
    """
    # Step size in pixels for spot_to_spot microns
    step_size_in_pixels = spot_to_spot * pixels_per_micron
    
    # Generate X-axis and Y-axis grid points
    X1 = np.arange(step_size_in_pixels, im.shape[1] - 2 * step_size_in_pixels, step_size_in_pixels * np.sqrt(3)/2)
    Y1 = np.arange(step_size_in_pixels, im.shape[0] - step_size_in_pixels, step_size_in_pixels)
    
    # Shift every other column by half a step size (for staggered pattern in columns)
    positions = []
    for i, x in enumerate(X1):
        if i % 2 == 0:  # Even columns (no shift)
            Y_shifted = Y1
        else:  # Odd columns (shifted by half)
            Y_shifted = Y1 + step_size_in_pixels / 2
        
        # Combine X and Y positions, and check for boundary conditions
        for y in Y_shifted:
            if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
                positions.append([x, y])
    
    return np.array(positions).T 

def create_disk_kernel(radius, shape):
    rr, cc = skimage.draw.disk((radius, radius), radius, shape=shape)
    kernel = np.zeros(shape, dtype=bool)
    kernel[rr, cc] = True
    return kernel

def apply_median_filter(image, kernel):
    return scipy.ndimage.median_filter(image, footprint=kernel)

def grid_anno(
    im,
    annotation_image_list,
    annotation_image_names,
    annotation_label_list,
    spot_to_spot,
    ppm_in,
    ppm_out,
):
    print(f'Generating grid with spacing - {spot_to_spot}, from annotation resolution of - {ppm_in} ppm')
    
    positions = generate_hires_grid(im, spot_to_spot, ppm_in).T  # Transpose for correct orientation
    
    radius = int(round((spot_to_spot / 2 ) * ppm_in)-1)
    kernel = create_disk_kernel(radius, (2 * radius + 1, 2 * radius + 1))

    df = pd.DataFrame(positions, columns=['x', 'y'])
    df['index'] = df.index

    for idx0, anno in enumerate(annotation_image_list):
        anno_orig = skimage.transform.resize(anno, im.shape[:2], preserve_range=True).astype('uint8')
        filtered_image = apply_median_filter(anno_orig, kernel)

        median_values = [filtered_image[int(point[1]), int(point[0])] for point in positions]
        anno_dict = {idx: annotation_label_list[idx0].get(val, "Unknown") for idx, val in enumerate(median_values)}
        number_dict = {idx: val for idx, val in enumerate(median_values)}

        df[annotation_image_names[idx0]] = list(anno_dict.values())
        df[annotation_image_names[idx0] + '_number'] = list(number_dict.values())

    df['x'] = df['x'] * ppm_out / ppm_in
    df['y'] = df['y'] * ppm_out / ppm_in
    df.set_index('index', inplace=True)

    return df


def dist2cluster_fast(df, annotation, KNN=5, logscale=False):
    from scipy.spatial import cKDTree

    print('calculating distance matrix with cKDTree')

    points = np.vstack([df['x'],df['y']]).T
    categories = np.unique(df[annotation])

    Dist2ClusterAll = {c: np.zeros(df.shape[0]) for c in categories}

    for idx, c in enumerate(categories):
        indextmp = df[annotation] == c
        if np.sum(indextmp) > KNN:
            print(c)
            cluster_points = points[indextmp]
            tree = cKDTree(cluster_points)
            # Get KNN nearest neighbors for each point
            distances, _ = tree.query(points, k=KNN)
            # Store the mean distance for each point to the current category
            if KNN == 1:
                Dist2ClusterAll[c] = distances # No need to take mean if only one neighbor
            else:
                Dist2ClusterAll[c] = np.mean(distances, axis=1)

    for c in categories:              
        if logscale:
            df["L2_dist_log10_"+annotation+'_'+c] = np.log10(Dist2ClusterAll[c])
        else:
            df["L2_dist_"+annotation+'_'+c] = Dist2ClusterAll[c]

    return Dist2ClusterAll

import numpy as np
import pandas as pd


def anno_to_cells(df_cells, x_col, y_col, df_grid, annotation='annotations', plot=True):
    """
    Maps tissue annotations to segmented cells by nearest neighbors.
    
    Parameters
    ----------
    df_cells : pandas.DataFrame
        Dataframe with cell data.
    x_col : str
        Name of column with x coordinates in df_cells.
    y_col : str
        Name of column with y coordinates in df_cells.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    annotation : str, optional
        Name of the column with annotations in df_grid. Default is 'annotations'.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the cell space to make sure 
        they are aligned. Default is True.

    Returns
    -------
    df_cells : pandas.DataFrame
        Updated dataframe with cell data.
    """
    
    print('make sure the coordinate systems are aligned e.g. axes are not flipped') 
    a = np.vstack([df_grid['x'], df_grid['y']])
    b = np.vstack([df_cells[x_col], df_cells[y_col]])
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('cell space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        df_grid_temp = df_grid.iloc[np.where(df_grid[annotation] != 'unassigned')[0], :].copy()
        aa = np.vstack([df_grid_temp['x'], df_grid_temp['y']])
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(aa[0], aa[1], '.', markersize=1)
        plt.title('annotation space')
        plt.show()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])]
    
    for k in annotations:
        print('migrating - ' + k + ' to segmentations')
        df_cells[k] = scipy.interpolate.griddata(points=a.T, values=df_grid[k], xi=b.T, method='nearest')
  
    return df_cells


def anno_to_visium_spots(df_spots, df_grid, ppm, plot=True,how='nearest',max_distance=10e10):
    """
    Maps tissue annotations to Visium spots according to the nearest neighbors.
    
    Parameters
    ----------
    df_spots : pandas.DataFrame
        Dataframe with Visium spot data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm : float 
        scale of annotation vs visium
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure 
        they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance : int
        maximal distance where points are not migrated 

    Returns
    -------
    df_spots : pandas.DataFrame
        Updated dataframe with Visium spot data.
    """
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    
    print('Make sure the coordinate systems are aligned, e.g., axes are not flipped.') 
    a = np.vstack([df_grid['x'], df_grid['y']])
    b = np.vstack([df_spots['pxl_col_in_fullres'], df_spots['pxl_row_in_fullres']])*ppm
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Spot space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('Morpho space')
        plt.show()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])].copy()
    
    for k in annotations:
        print('Migrating - ' + k + ' to segmentations.')
              
        # Interpolation
        df_spots[k] = griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        # Mask df_spots where the distance is too high
        df_spots[k][distances==np.inf] = None
        # df_spots[k] = scipy.interpolate.griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
  
    return df_spots


def plot_grid(df, annotation, spotsize=10, save=False, dpi=100, figsize=(5,5), savepath=None):
    """
    Plots a grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to be plotted.
    annotation : str
        Annotation to be used in the plot.
    spotsize : int, optional
        Size of the spots in the plot. Default is 10.
    save : bool, optional
        If true, saves the plot. Default is False.
    dpi : int, optional
        Dots per inch for the plot. Default is 100.
    figsize : tuple, optional
        Size of the figure. Default is (5,5).
    savepath : str, optional
        Path to save the plot. Default is None.

    Returns
    -------
    None
    """

    plt.figure(dpi=dpi, figsize=figsize)

    ct_order = list((df[annotation].value_counts() > 0).keys())
    ct_color_map = dict(zip(ct_order, np.array(sns.color_palette("colorblind", len(ct_order)))[range(len(ct_order))]))

    sns.scatterplot(x='x', y='y', hue=annotation, s=spotsize, data=df, palette=ct_color_map, hue_order=ct_order)

    plt.grid(False)
    plt.title(annotation)
    plt.axis('equal')

    if save:
        if savepath is None:
            raise ValueError('The savepath must be specified if save is True.')

        plt.savefig(savepath + '/' + annotation.replace(" ", "_") + '.pdf')

    plt.show()


def find_files(directory, query):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if query in file:
                return os.path.join(root, file)



def anno_transfer(df_spots, df_grid, ppm_spots, ppm_grid, plot=True, how='nearest', max_distance=10e10):
    """
    Maps tissue annotations to Visium spots according to the nearest neighbors.
    
    Parameters
    ----------
    df_spots : pandas.DataFrame
        Dataframe with Visium spot data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm_spots : float 
        pixels per micron of spots
    ppm_grid : float 
        pixels per micron of grid
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure 
        they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance : int
        maximal distance where points are not migrated 

    Returns
    -------
    df_spots : pandas.DataFrame
        Updated dataframe with Visium spot data.
    """
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt
    
    print('Make sure the coordinate systems are aligned, e.g., axes are not flipped.') 
    a = np.vstack([df_grid['x']/ppm_grid, df_grid['y']/ppm_grid])
    b = np.vstack([df_spots['x']/ppm_spots, df_spots['y']/ppm_spots])
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Spot space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('Morpho space')
        plt.show()

    # Create new DataFrame
    new_df_spots = df_spots[['x', 'y']].copy()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])].copy()
    
    for k in annotations:
        print('Migrating morphology - ' + k + ' to target space.')
        
        # Interpolation
        new_df_spots[k] = griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        
        # Mask df_spots where the distance is too high
        new_df_spots[k][distances==np.inf] = None
  
    return new_df_spots



def anno_to_grid(folder, file_name, spot_to_spot):
    """
    Load annotations and transform them into a spot grid, output is always in micron space to make sure distance calculations are correct,
    or in other words ppm=1.
    
    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    spot_to_spot : float
        The distance in microns used for grid spacing.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the grid annotations.
    """
    
    annotation_object, ppm = load_annotation(folder, file_name)

    df = grid_anno(
        annotation_object.label_image,
        [annotation_object.label_image,],
        [file_name],
        [{i+1: v for i, v in enumerate(annotation_object.annotation_map.keys())}],
        spot_to_spot,
        ppm,
        1,
    )

    return df


def map_annotations_to_visium(vis_path, df_grid, ppm_grid, spot_to_spot, plot=True, how='nearest', max_distance_factor=50, use_resolution='hires', res_in_ppm=1, count_file='raw_feature_bc_matrix.h5'):
    """
    Processes Visium data with high-resolution grid.

    Parameters
    ----------
    vis_path : str
        Path to the Visium data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm_grid : float 
        Pixels per micron of grid.
    spot_to_spot : float
        Spacing of the spots.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance_factor : int
        Factor to calculate maximal distance where points are not migrated. The final max_distance used will be max_distance_factor * ppm_visium.
    use_resolution : str, optional
        Resolution to use. Default is 'hires'.
    res_in_ppm : float, optional
        Resolution in pixels per micron. Default is 1.
    count_file : str, optional
        Filename of the count file. Default is 'raw_feature_bc_matrix.h5'.

    Returns
    -------
    adata_vis : anndata.AnnData
        Annotated data matrix with Visium spot data and additional annotations.
    """
    import scanpy as sc
    # calculate distance matrix between hires and visium spots
    im, ppm_visium, visium_positions = read_visium(spaceranger_dir_path=vis_path+'/', use_resolution=use_resolution, res_in_ppm=res_in_ppm, plot=False)
    
    # rename columns for visium_positions DataFrame
    visium_positions.rename(columns={'pxl_row_in_fullres': "y", 'pxl_col_in_fullres': "x"}, inplace=True) 

    # Transfer annotations
    spot_annotations = anno_transfer(df_spots=visium_positions, df_grid=df_grid, ppm_spots=ppm_visium, ppm_grid=ppm_grid, plot=plot, how=how, max_distance=max_distance_factor * ppm_visium)

    # Read visium data
    adata_vis = sc.read_visium(vis_path, count_file=count_file)

    # Merge with adata_vis
    adata_vis.obs = pd.concat([adata_vis.obs, spot_annotations], axis=1)

    # Convert to int
    adata_vis.obsm['spatial'] = adata_vis.obsm['spatial'].astype('int')

    # Add to uns
    adata_vis.uns['hires_grid'] = df_grid
    adata_vis.uns['hires_grid_ppm'] = ppm_grid
    adata_vis.uns['hires_grid_diam'] = spot_to_spot
    adata_vis.uns['visium_ppm'] = ppm_visium



    return adata_vis


def load_and_combine_annotations(folder, file_names, spot_to_spot, load_colors=True):
    """
    Load tissue annotations from multiple files and combine them into a single DataFrame.

    Parameters
    ----------
    folder : str
        Folder path where the annotation files are stored.
    file_names : list of str
        List of names of the annotation files.
    spot_to_spot : int
        spacing of the spots.
    load_colors : bool, optional
        Whether to load colors. Default is True.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame that combines all the loaded annotations.
    ppm_grid : float
        Pixels per micron for the grid of the last loaded annotation.
    """
    df_list = []
    ppm_grid = None

    for file_name in file_names:
        df, ppm_grid = anno_to_grid(folder=folder, file_name=file_name, spot_to_spot=spot_to_spot, load_colors=load_colors)
        df_list.append(df)

    # Concatenate all dataframes
    df = pd.concat(df_list, join='inner', axis=1)

    # Remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df, ppm_grid

def annotate_l2(df_target, df_grid, annotation='annotations_level_0', KNN=10, max_distance_factor=50, plot=False,calc_dist=True):
    """
    Process the given AnnData object to calculate distances and perform annotation transfer.
    
    Parameters
    ----------
    df_target : pandas.DataFrame
        Dataframe with target data to be annotated.
    df_grid : pandas.DataFrame
        Dataframe with annotation data.
    annotation : str, optional
        Annotation column to be used for calculating distances, by default 'annotations_level_0'.
    ppm_grid : float
        should be the ppm resolution fot the grid data for visium would be stored here - adata_vis.uns['hires_grid_ppm']
    ppm_spots : float
        should be the ppm resolution fot the target data for visium would be stored here - adata_vis.uns['visium_ppm']
    KNN : int, optional
        Number of nearest neighbors to be considered for distance calculation, by default 10.
    max_distance_factor : int, optional
        Factor by which to calculate the maximum distance for annotation transfer, by default 50 in microns.
    plot : bool, optional
        Whether to plot during the annotation transfer, by default False.
     calc_dist : bool, optional
        If true, calculates the L2 distance. Default is True otherwise just migrates discrete annotations.
    
    Returns
    -------
    anndata.AnnData
        Processed AnnData object with updated observations.
    """
    
    df = df.obs[['x','y']].dropna()
    dist2cluster_fast(df_grid, annotation=annotation, KNN=KNN,calc_dist=calc_dist) # calculate minimum mean distance of each spot to clusters 
    df_grid_new = df_grid.filter(like=annotation)
    df_grid_new['x'] = df_grid['x']
    df_grid_new['y'] = df_grid['y']
    spot_annotations = anno_transfer(df_spots=df_visium, df_grid=df_grid_new, ppm_spots=ppm_spots, ppm_grid=adata_vis.uns['hires_grid_ppm'], max_distance=max_distance_factor * adata_vis.uns['visium_ppm'], plot=plot)
    for col in spot_annotations:
        adata_vis.obs[col] = spot_annotations[col]
        adata_vis.uns['hires_grid'][col] = df_grid_new[col]
    # df1 = pd.concat([adata_vis.obs, spot_annotations], axis=1)
    # df1 = df1.loc[:, ~df1.columns.duplicated()].copy()
    # adata_vis.obs = df1

    return adata_vis

from scipy.spatial import cKDTree

def map_annotations_to_target(df_source, df_target, ppm_target, ppm_source=1, plot=True, max_distance=50):
    """
    Map annotations from a source to a target DataFrame based on nearest neighbor matching within a maximum distance.

    Parameters
    ----------
    df_source : pandas.DataFrame
        DataFrame with grid data and annotations.
    df_target : pandas.DataFrame
        DataFrame with target data.
    ppm_source : float 
        Pixels per micron of source data.
    ppm_target : float 
        Pixels per micron of target data.
    plot : bool, optional
        If True, plots the coordinates of the grid space and the spot space to verify alignment. Default is True.
    max_distance : int
        Maximum allowable distance for matching points. Final max_distance used will be max_distance * ppm_target.
   
    Returns
    -------
    df_target : pandas.DataFrame
        Annotated DataFrame with additional annotations from the source data.
    """

    # Adjust coordinate scaling
    a = np.vstack([df_source['x'] / ppm_source, df_source['y'] / ppm_source]).T
    b = np.vstack([df_target['x'] / ppm_target, df_target['y'] / ppm_target]).T
    
    # Plot the coordinate spaces if requested, overlaying them in a single plot with different colors and a legend
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.scatter(a[:, 0], a[:, 1], s=5, color='blue', label='Source Space', alpha=0.5)
        plt.scatter(b[:, 0], b[:, 1], s=5, color='orange', label='Target Space', alpha=0.5)
        plt.title('Source and Target Space Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()

    # Find nearest neighbors and distances only once
    tree = cKDTree(a)
    distances, indices = tree.query(b, distance_upper_bound=max_distance * ppm_target)
    
    # Filter valid indices based on distance and within-bounds check
    valid_mask = (indices < len(df_source)) & (distances < max_distance * ppm_target)
    
    # For each annotation, assign the value from the nearest neighbor in the source data
    annotations = df_source.columns.difference(['x', 'y'])
    for k in annotations:
        # Initialize with NaN or None where indices are out of bounds
        if pd.api.types.is_numeric_dtype(df_source[k]):
            df_target[k] = np.nan
        else:
            df_target[k] = None

        # Assign values where distance criteria are met and indices are valid
        valid_indices = indices[valid_mask]
        df_target.loc[valid_mask, k] = df_source.iloc[valid_indices][k].values

    return df_target


def read_visium_table(vis_path):
    """
    This function reads a scale factor from a json file and a table from a csv file, 
    then calculates the 'ppm' value and returns the table with the new column names.
    
    note that for CytAssist the header is changes so this funciton should be updated
    """
    with open(vis_path + '/spatial/scalefactors_json.json', 'r') as f:
        scalef = json.load(f)

    ppm = scalef['spot_diameter_fullres'] / 55 

    df_visium_spot = pd.read_csv(vis_path + '/spatial/tissue_positions_list.csv', header=None)

    df_visium_spot.rename(columns={4:'y',5:'x',1:'in_tissue',0:'barcode'}, inplace=True)
    df_visium_spot.set_index('barcode', inplace=True)

    return df_visium_spot, ppm


def calculate_axis_3p(df_ibex, anno, structure, output_col, w=[0.5,0.5], prefix='L2_dist_'):
    """
    Function to calculate a unimodal nomralized axis based on ordered structure of S1 -> S2 -> S3.

    Parameters:
    -----------
    df_ibex : DataFrame
        Input DataFrame that contains the data.
    anno : str, optional
        Annotation column. 
    structure : list of str, optional
        List of structures to be meausure. [S1, S2, S3]
    w : list of float, optional
        List of weights between the 2 components of the axis w[0] * S1->S2 and w[1] * S2->S3. Default is [0.2,0.8].
    prefix : str, optional
        Prefix for the column names in DataFrame. Default is 'L2_dist_'.
    output_col : str, optional
        Name of the output column.

    Returns:
    --------
    df : DataFrame
        DataFrame with calculated new column.
    """
    df = df_ibex.copy()
    a1 = (df[prefix + anno +'_'+ structure[0]] - df[prefix + anno +'_'+ structure[1]]) \
    /(df[prefix + anno +'_'+ structure[0]] + df[prefix + anno +'_'+ structure[1]])
    
    a2 = (df[prefix + anno +'_'+ structure[1]] - df[prefix + anno +'_'+ structure[2]]) \
    /(df[prefix + anno +'_'+ structure[1]] + df[prefix + anno +'_'+ structure[2]])
    df[output_col] = w[0]*a1 + w[1]*a2
    
    return df


def calculate_axis_2p(df_ibex, anno, structure, output_col, prefix='L2_dist_'):
    """
    Function to calculate a unimodal nomralized axis based on ordered structure of S1 -> S2 .

    Parameters:
    -----------
    df_ibex : DataFrame
        Input DataFrame that contains the data.
    anno : str, optional
        Annotation column. 
    structure : list of str, optional
        List of structures to be meausure. [S1, S2]
    prefix : str, optional
        Prefix for the column names in DataFrame. Default is 'L2_dist_'.
    output_col : str, optional
        Name of the output column.

    Returns:
    --------
    df : DataFrame
        DataFrame with calculated new column.
    """
    df = df_ibex.copy()
    a1 = (df[prefix + anno +'_'+ structure[0]] - df[prefix + anno +'_'+ structure[1]]) \
    /(df[prefix + anno +'_'+ structure[0]] + df[prefix + anno +'_'+ structure[1]])

    df[output_col] = a1 
    
    return df

def bin_axis(ct_order, cutoff_values, df, axis_anno_name):
    """
    Bins a column of a DataFrame based on cutoff values and assigns manual bin labels.

    Parameters:
        ct_order (list): The order of manual bin labels.
        cutoff_values (list): The cutoff values used for binning.
        df (pandas.DataFrame): The DataFrame containing the column to be binned.
        axis_anno_name (str): The name of the column to be binned.

    Returns:
        pandas.DataFrame: The modified DataFrame with manual bin labels assigned.
    """
    # Manual annotations
    df['manual_bin_' + axis_anno_name] = 'unassigned'
    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('object')
    df.loc[np.array(df[axis_anno_name] < cutoff_values[0]), 'manual_bin_' + axis_anno_name] = ct_order[0]
    print(ct_order[0] + '= (' + str(cutoff_values[0]) + '>' + axis_anno_name + ')')
    
    for idx, r in enumerate(cutoff_values[:-1]):
        df.loc[np.array(df[axis_anno_name] >= cutoff_values[idx]) & np.array(df[axis_anno_name] < cutoff_values[idx+1]),
               'manual_bin_' + axis_anno_name] = ct_order[idx+1]
        print(ct_order[idx+1] + '= (' + str(cutoff_values[idx]) + '<=' + axis_anno_name + ') & (' + str(cutoff_values[idx+1]) + '>' + axis_anno_name + ')' )

    df.loc[np.array(df[axis_anno_name] >= cutoff_values[-1]), 'manual_bin_' + axis_anno_name] = ct_order[-1]
    print(ct_order[-1] + '= (' + str(cutoff_values[-1]) + '=<' + axis_anno_name + ')')

    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('category')
    df['manual_bin_' + axis_anno_name + '_int'] = df['manual_bin_' + axis_anno_name].cat.codes

    return df


def plot_cont(data, x_col='centroid-1', y_col='centroid-0', color_col='L2_dist_annotation_tissue_Edge', 
               cmap='jet', title='L2_dist_annotation_tissue_Edge', s=1, dpi=100, figsize=[10,10]):
    plt.figure(dpi=dpi, figsize=figsize)

    # Create an axes instance for the scatter plot
    ax = plt.subplot(111)

    # Create the scatterplot
    scatter = sns.scatterplot(x=x_col, y=y_col, data=data, 
                              c=data[color_col], cmap=cmap, s=s, 
                              legend=False, ax=ax)  # Use the created axes

    plt.grid(False)
    plt.axis('equal')
    plt.title(title)
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

    # Add colorbar
    norm = plt.Normalize(data[color_col].min(), data[color_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=title, aspect=30)  # Use the created axes for the colorbar
    cbar.ax.set_position([0.85, 0.25, 0.05, 0.5])  # adjust the position as needed
   

    plt.show()

