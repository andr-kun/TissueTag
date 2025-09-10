import json

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

import tissue_tag.io
from tissue_tag import read_visium


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
    # calculate distance matrix between hires and visium spots
    im, ppm_visium, visium_positions = read_visium(spaceranger_dir_path=vis_path + '/', use_resolution=use_resolution, ppm_out=res_in_ppm, plot=False)

    # rename columns for visium_positions DataFrame
    visium_positions.rename(columns={'pxl_row_in_fullres': "y", 'pxl_col_in_fullres': "x"}, inplace=True)

    # Transfer annotations
    spot_annotations = anno_transfer(df_spots=visium_positions, df_grid=df_grid, ppm_spots=ppm_visium, ppm_grid=ppm_grid, plot=plot, how=how, max_distance=max_distance_factor * ppm_visium)

    # Read visium data
    adata_vis = tissue_tag.io.read_visium(vis_path, count_file=count_file)

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
