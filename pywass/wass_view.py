#!/usr/bin/env python3
"""
WASS output viewer script for visual quality control.
"""
import os 
import sys 
import glob 
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
from pywass import wass_load as wlo

# Main script
if __name__ == '__main__':
    from argparse import ArgumentParser
    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data_root (work) directory"),
                type=str,
                default='/home/mikapm/Github/wass/test/ekok_20220916',
                )
        parser.add_argument("-date", 
                help=("Viewing period start date+time, yyyymmddHHMM"),
                type=str,
                )
        parser.add_argument("--imgrid", 
                help=("Also plot imgrids (must be available)?"),
                action='store_true'
                )
        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Directories etc.
    datestr = args.date[:8]
    timestr = args.date[8:]
    expdir = os.path.join(args.dr, datestr, timestr)

    # Initialize WASS_load object
    WL = wlo.WASS_load(expdir,)
    # List available ncfiles
    griddir = os.path.join(WL.data_root, 'grid')
    fns_xyg = sorted(glob.glob(os.path.join(griddir, 'xygrid_*.nc')))
    # Also list imgrid files if requested
    if args.imgrid:
        fns_img = sorted(glob.glob(os.path.join(griddir, 'imgrid_*.nc')))
    # If more than one netcdf file, read and concatenate into one dataset
    if len(fns_xyg) > 1:
        dsz_list = [] # List for concatenating
        for fn in fns_xyg:
            # Read file and append to list for concatenating
            dsz = xr.open_dataset(fn, decode_coords='all')
            dsz_list.append(dsz)
        # Concatenate datasets on time coordinate
        dsz = xr.concat(dsz_list, dim='time')
        # Also read and concatenate imgrids?
        if args.imgrid:
            for fn in fns_img:
                dsi_list = [] # List for concatenating
                dsi = xr.open_dataset(fn, decode_coords='all')
                dsi_list.append(dsi)
            # Concatenate imgrid datasets on time coordinate
            dsi = xr.concat(dsi_list, dim='time')
                