#!/usr/bin/env python3

"""
Filtering functions for gridded WASS output.
"""

import numpy as np 
import pandas as pd
import xarray as xr 
from scipy.ndimage import gaussian_filter
from datetime import datetime as DT

def interpolate_time(ds, fs=5):
    """
    Linearly interpolate over missing frames in WASS 
    xr.Dataset ds.

    Note: Changes the input dataset!

    Parameters
        ds - xr.Dataset with WASS eta(t,x,y) grid
        fs - scalar; sampling rate (Hz)

    Returns
        ds - Interpolated dataset
    """

    # Save grids before changing dataset
    xgrid = ds.xgrid
    ygrid = ds.ygrid
    # Interpolate in time
    print('Interpolating in time ...')
    ds = ds.sortby('time') # Sort by time just in case
    # Get record start and end time stamps
    t0 = DT.strptime(str(ds.time[0].values), '%Y-%m-%dT%H:%M:%S.%f000')
    t1 = DT.strptime(str(ds.time[-1].values), '%Y-%m-%dT%H:%M:%S.%f000')
    # Make new, full time index and interpolate
    ms = int((1/fs) * 1000) # Sampling rate in milliseconds
    tt0 = pd.Timestamp(t0).round(f'{ms}ms')
    tt1 = pd.Timestamp(t1).round(f'{ms}ms')
    t_new = pd.date_range(tt0, tt1, freq=f'{ms}ms')
    ds = ds.eta.interp(time=t_new).to_dataset()
    ds = ds.assign(xgrid=xgrid)
    ds = ds.assign(ygrid=ygrid)

    return ds


def filt_gauss3d(ds, fs=5, sigma=1.5, interpolate=False):
    """
    Apply 3D Gaussian filter to WASS xr.Dataset ds.

    Note: Changes the input dataset!

    Parameters
        ds - xr.Dataset with WASS eta(t,x,y) grid
        fs - scalar; sampling rate (Hz)
        sigma - scalar; sigma parameter for
                scipy.ndimage.gaussian_filter 
        interpolate - bool; if True, linearly interpolate 
                      over time dimension before filtering.
    
    Returns
        ds - Filtered (and interpolated) dataset
    """
    # Interpolate first if requested
    if interpolate:
        ds = interpolate_time(ds, fs=fs)
    # Compute radius parameter from sigma
    radius = np.ceil(2 * sigma).astype(int)
    print('Gaussian filtering in time + space ...')
    # print(f'sigma: {sigma:.2f}, radius: {radius:.2f}')
    res = gaussian_filter(ds.eta.values, sigma=sigma, radius=radius)
    # New dataset for filtered grids
    ds = xr.Dataset(data_vars={'eta':(['time','x','y'], res)},
                    coords={'time': (['time'], ds.time.values),
                            'x': (['x'], ds.x.values),
                            'y': (['y'], ds.y.values),
                            },
                    )#.dropna(dim='time', how='all') 

    return ds 


def filt_fft3d(ds, fs=5, np=4, interpolate=False):
    """
    Apply FFT-based 3D filter to xr.Dataset ds.
    Filters out frequencies > np*peak frequency.

    Based on filter_stereodata.m function by
    Francesco Fedele (GA Tech).

    Note: Changes the input dataset!

    Parameters
        ds - xr.Dataset with WASS eta(t,x,y) grid
        fs - scalar; sampling rate (Hz)
        np - scalar; multiple of peak freq. for cutoff
        interpolate - bool; if True, linearly interpolate 
                      over time dimension before filtering.
    
    Returns
        ds - Filtered (and interpolated) dataset
    """
    # Interpolate first if requested
    if interpolate:
        ds = interpolate_time(ds, fs=fs)
