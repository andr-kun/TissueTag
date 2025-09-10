import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import polygon
import skimage.transform
import skimage.draw
import scipy.ndimage
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def generate_hires_grid(im, spot_to_spot, pixels_per_micron):
    """
    Creates a hexagonal grid of a specified size and density.
    
    Parameters
    ----------
    im : numpy.ndarray
        Image to fit the grid on (mostly for dimensions).
    spot_to_spot : float
        Spot spacing to determine the grid density.
    pixels_per_micron : float
        The resolution of the image in pixels per micron.

    Returns
    -------
    numpy.ndarray
        Hexagonal grid coordinates
    """

    # Step size in pixels for spot_to_spot microns
    step_size_in_pixels = spot_to_spot * pixels_per_micron

    # Generate X-axis and Y-axis grid points
    X1 = np.arange(step_size_in_pixels, im.shape[1] - 2 * step_size_in_pixels, step_size_in_pixels * np.sqrt(3) / 2)
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
    """
    Create a disk-shaped kernel for filtering.

    Parameters
    ----------
    radius : int
        Radius of the disk.
    shape : tuple
        Shape of the kernel (height, width).

    Returns
    -------
    numpy.ndarray
        Disk-shaped kernel.
    """

    rr, cc = skimage.draw.disk((radius, radius), radius, shape=shape)
    kernel = np.zeros(shape, dtype=bool)
    kernel[rr, cc] = True
    return kernel


def generate_grid_from_annotation(
        tissue_tag_annotation,
        spot_to_spot,
        ppm_out=1,
        annotation_column='annotation'
):
    """
    Generate a grid and assign annotation values to each grid point based on the median value of the annotation image.

    Parameters
    ----------
    tissue_tag_annotation : TissueTagAnnotation
        TissueTagAnnotation object containing label_image
    spot_to_spot : float
        Spot spacing to determine the grid density.
    ppm_out : float
        The resolution of the output grid in pixels per micron.
    annotation_column : str, optional
        Column name for the annotation values. Default is 'annotation'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the grid coordinates and corresponding annotation values.
    """

    print(
        f'Generating grid with spacing - {spot_to_spot}, from annotation resolution of - {tissue_tag_annotation.ppm} ppm')

    positions = generate_hires_grid(tissue_tag_annotation.label_image, spot_to_spot,
                                    tissue_tag_annotation.ppm).T  # Transpose for correct orientation

    radius = int(round((spot_to_spot / 2) * tissue_tag_annotation.ppm) - 1)
    kernel = create_disk_kernel(radius, (2 * radius + 1, 2 * radius + 1))

    df = pd.DataFrame(positions, columns=['x', 'y'])
    df['index'] = df.index

    #for idx0, anno in enumerate(annotation_image_list):
    anno_orig = skimage.transform.resize(tissue_tag_annotation.label_image, tissue_tag_annotation.label_image.shape[:2],
                                         preserve_range=True).astype('uint8')
    filtered_image = scipy.ndimage.median_filter(anno_orig, footprint=kernel)

    median_values = [filtered_image[int(point[1]), int(point[0])] for point in positions]
    annotation_label_list = {i + 1: v for i, v in enumerate(tissue_tag_annotation.annotation_map.keys())}
    anno_dict = {idx: annotation_label_list.get(val, "Unknown") for idx, val in enumerate(median_values)}
    number_dict = {idx: val for idx, val in enumerate(median_values)}

    df[annotation_column] = list(anno_dict.values())
    df[annotation_column + '_number'] = list(number_dict.values())

    df['x'] = df['x'] * ppm_out / tissue_tag_annotation.ppm
    df['y'] = df['y'] * ppm_out / tissue_tag_annotation.ppm
    df.set_index('index', inplace=True)

    return df


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
    pandas.DataFrame
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


def calculate_distance_to_annotations(grid, knn=5, logscale=False, annotation_column='annotation', copy=False):
    """
    Calculate the nearest distance for each grid points to all annotation categories..

    Parameters
    ----------
    grid : pandas.DataFrame
        DataFrame containing the grid coordinates and annotations.
    knn : int, optional
        Number of nearest neighbors to consider. Default is 5.
    logscale : bool, optional
        Use logarithmic scale (base 10) for distances. Default is False.
    annotation_column : str, optional
        Column name for the annotation values within the grid dataframe. Default is 'annotation'.
    copy : bool, optional
        Return a new copy of the grid with the distance instead of modifying it in place. Default is False.

    Returns
    -------
    None | pandas.DataFrame
        DataFrame containing the grid coordinates and distances to each annotation category if copy is True,
        otherwise None.
    """

    grid = grid.copy() if copy else grid
    print('calculating distance matrix')

    points = np.vstack([grid['x'], grid['y']]).T
    categories = np.unique(grid[annotation_column])

    dist_to_annotations = {c: np.zeros(grid.shape[0]) for c in categories}

    for idx, c in enumerate(categories):
        indextmp = grid[annotation_column] == c
        if np.sum(indextmp) > knn:
            print(c)
            cluster_points = points[indextmp]
            tree = cKDTree(cluster_points)
            # Get KNN nearest neighbors for each point
            distances, _ = tree.query(points, k=knn)
            # Store the mean distance for each point to the current category
            if knn == 1:
                dist_to_annotations[c] = distances  # No need to take mean if only one neighbor
            else:
                dist_to_annotations[c] = np.mean(distances, axis=1)

    for c in categories:
        if logscale:
            grid["L2_dist_log10_" + annotation_column + '_' + c] = np.log10(dist_to_annotations[c])
        else:
            grid["L2_dist_" + annotation_column + '_' + c] = dist_to_annotations[c]

    print(dist_to_annotations)

    return grid if copy else None


def calculate_axis_3p(grid, structure, output_col, annotation_column='annotation', w=[0.5, 0.5], copy=False):
    """
    Function to calculate a uni-modal normalized axis based on ordered structure of S1 -> S2 -> S3.

    Parameters:
    -----------
    grid : pandas.DataFrame
        DataFrame containing the grid coordinates and annotations.
    structure : list of str
        List of structures to be measured. [S1, S2, S3]
    output_col : str
        Name of the output column.
    annotation_column : str, optional
        Column name for the annotation values within the grid dataframe. Default is 'annotation'.
    w : list of float, optional
        List of weights between the 2 components of the axis w[0] * S1->S2 and w[1] * S2->S3. Default is [0.2,0.8].
    copy : bool, optional
        Return a new copy of the grid with the distance instead of modifying it in place. Default is False.

    Returns:
    --------
    None | pandas.DataFrame
        DataFrame with calculated axis values if copy is True, otherwise None.
    """

    grid = grid.copy() if copy else grid
    prefix = 'L2_dist_'

    a1 = (grid[prefix + annotation_column + '_' + structure[0]] - grid[prefix + annotation_column + '_' + structure[1]]) \
         / (grid[prefix + annotation_column + '_' + structure[0]] + grid[prefix + annotation_column + '_' + structure[1]])

    a2 = (grid[prefix + annotation_column + '_' + structure[1]] - grid[prefix + annotation_column + '_' + structure[2]]) \
         / (grid[prefix + annotation_column + '_' + structure[1]] + grid[prefix + annotation_column + '_' + structure[2]])
    grid[output_col] = w[0] * a1 + w[1] * a2

    return grid if copy else None


def calculate_axis_2p(grid, structure, output_col, annotation_column='annotation', copy=False):
    """
    Function to calculate a uni-modal normalized axis based on ordered structure of S1 -> S2 .

    Parameters:
    -----------
     grid : pandas.DataFrame
        DataFrame containing the grid coordinates and annotations.
    structure : list of str
        List of structures to be measured. [S1, S2, S3]
    output_col : str
        Name of the output column.
    annotation_column : str, optional
        Column name for the annotation values within the grid dataframe. Default is 'annotation'.
    copy : bool, optional
        Return a new copy of the grid with the distance instead of modifying it in place. Default is False.

    Returns:
    --------
    None | pandas.DataFrame
        DataFrame with calculated axis values if copy is True, otherwise None.
    """

    grid = grid.copy() if copy else grid
    prefix = 'L2_dist_'
    a1 = (grid[prefix + annotation_column + '_' + structure[0]] - grid[prefix + annotation_column + '_' + structure[1]]) \
         / (grid[prefix + annotation_column + '_' + structure[0]] + grid[prefix + annotation_column + '_' + structure[1]])

    grid[output_col] = a1

    return grid if copy else None


def bin_axis(ct_order, cutoff_values, df, axis_anno_name):
    """
    Bins a column of a DataFrame based on cutoff values and assigns manual bin labels.

    Parameters:
    -----------
    ct_order : list of str
        The order of manual bin labels.
    cutoff_values : list of float
        The cutoff values used for binning.
    df : pandas.DataFrame
        The DataFrame containing the column to be binned.
    axis_anno_name : str
        The name of the column to be binned.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with manual bin labels assigned.
    """

    # Manual annotations
    df['manual_bin_' + axis_anno_name] = 'unassigned'
    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('object')
    df.loc[np.array(df[axis_anno_name] < cutoff_values[0]), 'manual_bin_' + axis_anno_name] = ct_order[0]
    print(ct_order[0] + '= (' + str(cutoff_values[0]) + '>' + axis_anno_name + ')')

    for idx, r in enumerate(cutoff_values[:-1]):
        df.loc[
            np.array(df[axis_anno_name] >= cutoff_values[idx]) & np.array(df[axis_anno_name] < cutoff_values[idx + 1]),
            'manual_bin_' + axis_anno_name] = ct_order[idx + 1]
        print(ct_order[idx + 1] + '= (' + str(cutoff_values[idx]) + '<=' + axis_anno_name + ') & (' + str(
            cutoff_values[idx + 1]) + '>' + axis_anno_name + ')')

    df.loc[np.array(df[axis_anno_name] >= cutoff_values[-1]), 'manual_bin_' + axis_anno_name] = ct_order[-1]
    print(ct_order[-1] + '= (' + str(cutoff_values[-1]) + '=<' + axis_anno_name + ')')

    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('category')
    df['manual_bin_' + axis_anno_name + '_int'] = df['manual_bin_' + axis_anno_name].cat.codes

    return df


def plot_cont(data, x_col='centroid-1', y_col='centroid-0', color_col='L2_dist_annotation_tissue_Edge',
              cmap='jet', title='L2_dist_annotation_tissue_Edge', s=1, dpi=100, figsize=[10, 10]):
    """
    Plot a scatter plot with color mapping based on a specified column.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to be plotted.
    x_col : str, optional
        Column name for x-axis. Default is 'centroid-1'.
    y_col : str, optional
        Column name for y-axis. Default is 'centroid-0'.
    color_col : str, optional
        Column name for color mapping. Default is 'L2_dist_annotation_tissue_Edge'.
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    title : str, optional
        Title of the plot. Default is 'L2_dist_annotation_tissue_Edge'.
    s : int, optional
        Size of the points in the scatter plot. Default is 1.
    dpi : int, optional
        Dots per inch for the figure. Default is 100.
    figsize : list of int, optional
        Size of the figure in inches. Default is [10, 10].

    Returns
    -------
    None
    """

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
