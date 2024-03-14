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
import hvplot.xarray
import panel as pn
from datetime import datetime as DT
import cmocean
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
        parser.add_argument("-i0", 
                help=("Viewing period start index"),
                type=int,
                default=0,
                )
        parser.add_argument("-i1", 
                help=("Viewing period end index"),
                type=int,
                default=20,
                )
        parser.add_argument("-xmin", 
                help=("Lower grid x limit."),
                type=float,
                default=None,
                )
        parser.add_argument("-xmax", 
                help=("Upper grid x limit."),
                type=float,
                default=None,
                )
        parser.add_argument("-ymin", 
                help=("Lower grid y limit. (y decreases outwards from platform)"),
                type=float,
                default=None,
                )
        parser.add_argument("-ymax", 
                help=("Upper grid y limit."),
                type=float,
                default=None,
                )
        parser.add_argument("--imgrid", 
                help=("Also plot imgrids (must be available)?"),
                action='store_true'
                )
        parser.add_argument("--raw", 
                help=("Also plot raw frames (from xygrid nc file)?"),
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
        print('Concatenating datasets ...')
        dsz_list = [] # List for concatenating
        for fn in fns_xyg:
            # Read file and append to list for concatenating
            dsz = xr.open_dataset(fn, decode_coords='all')
            dsz_list.append(dsz)
        # Concatenate datasets on time coordinate
        dsz = xr.concat(dsz_list, dim='time')
        # Select requested time slice
        dsz = dsz.isel(time=slice(args.i0, args.i1))
        # Zoom in if requested
        dsz = dsz.sel(X=slice(args.xmin, args.xmax), Y=slice(args.ymin, args.ymax))
        # Also read and concatenate imgrids?
        if args.imgrid:
            dsi_list = [] # List for concatenating
            for fn in fns_img:
                dsi = xr.open_dataset(fn, decode_coords='all')
                dsi_list.append(dsi)
            # Concatenate imgrid datasets on time coordinate
            dsi = xr.concat(dsi_list, dim='time')
            # Select requested time slice
            dsi = dsi.isel(time=slice(args.i0, args.i1))
            # Zoom in if requested
            dsi = dsi.sel(X=slice(args.xmin, args.xmax), Y=slice(args.ymin, args.ymax))
    
    # Make interactive plot using hvplot
    xyg = dsz.Z.hvplot.image(width=500, 
                             height=400,
                             clim=(dsz.Z.min().item(),dsz.Z.max().item()), 
                             cmap='coolwarm', 
                             dynamic=True,
                             widget_type='scrubber', 
                             widget_location='bottom',
                             )
    # Plot raw images?
    if args.raw:
        raw = dsz.cam0images.hvplot.image(x='ih', y='iw',
                                        width=500, 
                                        height=400, 
                                        clim=(0,255),
                                        widget_type='scrubber',
                                        widget_location='bottom',
                                        cmap='gray', 
                                        dynamic=True,
                                        )
    # Plot reprojected image grids?
    if args.imgrid:
        img = dsi.im_grid.hvplot.image(dynamic=True, 
                                       width=500, 
                                       height=400,
                                       clim=(0,255),
                                       cmap='gray',
                                       widget_type='scrubber',
                                       widget_location='bottom',
                                       )
    if not args.raw and not args.imgrid:
        hvplot.show(xyg)
    elif args.raw and not args.imgrid:
        hvplot.show(xyg + raw)
    elif not args.raw and args.imgrid:
        hvplot.show(xyg + img)
    elif args.raw and args.imgrid:
        hvplot.show(xyg + img + raw)

                