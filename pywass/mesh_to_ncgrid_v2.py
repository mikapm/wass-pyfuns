"""
Updated WASS point cloud gridding script. Designed to work (with minor adjustments)
with wassncplot library.

Use 'wass' conda environment from wass-pyfuns library:
    conda activate wass

Grid limits for different Newport trials:
    python grid_mesh.py -exp 1 -xmin -75 -xmax 45 -ymin -200 -ymax -50 -baseline 6.68
    python grid_mesh.py -exp 2 -xmin -70 -xmax 50 -ymin -200 -ymax -45 -baseline 6.68
    python grid_mesh.py -exp 3 -xmin -40 -xmax 20 -ymin -100 -ymax -40 -baseline 2.58
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from cftime import date2num
import cmocean.cm as cmo
from argparse import ArgumentParser
from pywass import wass_load as wlo
from pywass.mesh_to_ncgrid import interp_weights

def parse_args(**kwargs):
    """
    Input arguments for main script
    """
    parser = ArgumentParser()

    parser.add_argument("-dr", 
            help=("WASS experiment root directory."),
            type=str,
            default='/home/mikapm/Github/wass/test/hopkins_20231001',
            )
    parser.add_argument("-exp", 
            help=("Experiment name for output .nc filename."),
            type=str,
            default='test1',
            )
    parser.add_argument("-fn_root", 
            help=("Start string for output .nc filename"),
            type=str,
            default='wass',
            )
    parser.add_argument("-dxy", 
            help=("Grid resolution (in x and y) in meters."),
            type=float,
            default=0.75,
            )
    parser.add_argument("-iw", 
            help=("Image width in pixels."),
            type=int,
            default=734,
            )
    parser.add_argument("-ih", 
            help=("Image height in pixels."),
            type=int,
            default=614,
            )
    parser.add_argument("-xmin", 
            help=("Lower grid x limit."),
            type=float,
            default=-50,
            )
    parser.add_argument("-xmax", 
            help=("Upper grid x limit."),
            type=float,
            default=60,
            )
    parser.add_argument("-ymin", 
            help=("Lower grid y limit. (y decreases outwards from platform)"),
            type=float,
            default=-160,
            )
    parser.add_argument("-ymax", 
            help=("Upper grid y limit."),
            type=float,
            default=-60,
            )
    parser.add_argument("-step", 
            help=("Near-field subsampling step. Set to 0 to skip subsampling."),
            type=int,
            default=3,
            )
    parser.add_argument("-plane", 
            help=("Which average plane to use."),
            type=str,
            default='sub',
            choices=['sub', 'pipe', 'all'],
            )
    parser.add_argument("-ref_date", 
            help=("Reference start date for netcdf timestamps."),
            type=str,
            default='2000-01-01',
            )
    parser.add_argument("-baseline", 
            help=("Stereo baseline [m]."),
            type=float,
            default=5.51,
            )
    parser.add_argument("-ser_cam0", 
            help=("Serial number of cam0."),
            type=str,
            default='1014929',
            )
    parser.add_argument("-ser_cam1", 
            help=("Serial number of cam1."),
            type=str,
            default='1014932',
            )
    parser.add_argument("-fillvalue", 
            help=("Fill value for netcdf file."),
            type=float,
            default=-9999.,
            )

    return parser.parse_args(**kwargs)

# Variables from input args
args = parse_args(args=sys.argv[1:])

# Directories etc.
rootdir = os.path.join(args.dr, args.exp)
outdir = os.path.join(rootdir, 'grid') # Output grid dir
if not os.path.isdir(outdir):
    # Make grid dir if it does not exist
    os.mkdir(outdir)
# Output netcdf filename
fn_nc = os.path.join(outdir, '{}_{}.nc'.format(args.fn_root, args.exp))

# Check if netcdf file already exists
if not os.path.isfile(fn_nc):
    # Initialize WASS_load object
    WL = wlo.WASS_load(rootdir, plane=args.plane, scale=args.baseline)
    # Load plane file for saving in output dataset
    planefile = os.path.join(WL.data_root,
            'plane_avg_{}.txt'.format(WL.plane))
    plane = np.loadtxt(planefile)

    # Use image filenames to get time array. Use cam1 timestamps by default.
    timestamps = [] # Empty list to store timestamps
    t0 = pd.Timestamp('1970-01-01 00:00:00') # Camera clock start time
    for fn in WL.in_cam1:
        sec = os.path.split(fn)[1].split('_')[1][:10] # seconds cam1
        ms = os.path.split(fn)[1].split('_')[1][10:] # milliseconds cam1
        # Construct timestamp
        ts = t0 + pd.Timedelta(seconds=int(sec)) + pd.Timedelta(milliseconds=int(ms))
        # Append to list
        timestamps.append(ts)

    # Construct pd.DataFrame for conversion of timestamps to numerical format
    df = pd.DataFrame(data=np.zeros(len(timestamps)), index=np.array(timestamps))

    # Convert time array to numerical format
    time_units = 'seconds since {:%Y-%m-%d 00:00:00}'.format(
            pd.Timestamp(args.ref_date))
    time_vals = date2num(df.index.to_pydatetime(), 
                         time_units, calendar='standard', 
                         has_year_zero=True)

    # Initialize common x,y grids
    xgrid, ygrid = np.mgrid[args.xmin:args.xmax+1:args.dxy, 
        args.ymin:args.ymax+1:args.dxy]
    # xy_grid needed for computing vertices and weights for interpolation
    xy_grid = np.vstack((xgrid.flatten(),ygrid.flatten())).T

    # Initialize output xr.Dataset for saving as netcdf
    nt = len(timestamps) # no. of timestamps
    # nx = int((args.xmax - args.xmin) / args.dxy + 1) # no. of x grid points
    ny = int((args.ymax - args.ymin) / args.dxy + 1) # no. of y grid points
    # xs = np.linspace(args.xmin, args.xmax, nx) # x array (1D)
    xs = xgrid[:,0] # x array (1D)
    nx = len(xs)
    # ys = np.linspace(args.ymin, args.ymax, ny) # y array (1D)
    ys = ygrid[0,:] # x array (1D)
    ny = len(ys)
    ds = xr.Dataset(
            data_vars=dict(
                Z = (['time', 'X', 'Y'], np.ones((nt, nx, ny))*np.nan),
                X_grid = (['X', 'Y'], xgrid),
                Y_grid = (['X', 'Y'], ygrid),
                cam0images = (['time', 'ih', 'iw'], np.ones((nt, args.ih, args.iw))*np.nan),
                scale = ([], args.baseline),
                p0plane = (['V4'], plane),
                ),
            coords=dict(
                time = (['time'], time_vals),
                X = (['X'], xs),
                Y = (['Y'], ys),
                V4 = (['V4'], np.arange(4)),
                ih = (['ih'], np.arange(args.ih)),
                iw = (['iw'], np.arange(args.iw)),
                ),
        )



    # Iterate over WASS output working directories (wd) and grid point clouds
    for i, wd in tqdm(enumerate(tqdm(WL.wd))):
        if os.path.isfile(os.path.join(wd, 'mesh_cam.xyzC')):
            # Load mesh
            mesh = WL.load_camera_mesh(idx=i)
            # Align with plane
            mesh_al, plane = WL.align_plane(mesh=mesh, return_plane=True)
            # Project mesh to cam1 view for pixel coordinates
            _, pt2d = WL.project_mesh_to_cam(i, plane, mesh)
            pt2d = pt2d.T # Transpose to same shape as mesh_al

            # Exclude points outside xmin, xmax & ymin, ymax
            x, y, z = mesh_al.T
            mesh_al = mesh_al[(x>args.xmin) & (x<args.xmax) & \
                    (y>args.ymin) & (y<args.ymax), :]
            pt2d = pt2d[(x>args.xmin) & (x<args.xmax) & (y>args.ymin) & (y<args.ymax),:]
            # x,y,z coordinates from (reduced) aligned mesh
            x, y, z = mesh_al.T

            if args.step > 0:
                # Subsample aligned mesh in the near-field for faster interpolation
                # following Grid_surfaces_wass.m
                print('Subsampling mesh for faster interpolation ... \n')
                dummy = (args.ymax - args.ymin) / args.step
                y_min = args.ymin - 0.1
                for step_d in range(1, args.step+2):
                    g = np.where(np.logical_and(y>=y_min, y<y_min+dummy))[0]
                    if step_d == 1:
                        mesh_subs = mesh_al[g[0::step_d]]
                        # Also subsample unaligned mesh for pixel coordinates
                        pt2d_subs = pt2d[g[0::step_d]]
                    else:
                        mesh_subs = np.vstack((mesh_subs, mesh_al[g[0::step_d]]))
                        # Also subsample unaligned mesh for pixel coordinates
                        pt2d_subs = np.vstack((pt2d_subs, pt2d[g[0::step_d]]))
                    y_min += dummy
            elif args.step == 0:
                # Don't subsample (slow)
                mesh_subs = mesh

            # Speed up gridding by calculating vertices and weights only once.
            x, y, z = mesh_subs.T
            xy = np.vstack((x.flatten(), y.flatten())).T 
            print('Computing vertices \n')
            vertices, weights = interp_weights(xy, xy_grid)

            # Grid mesh to eta grid and pixel coordinates to iR and jR
            print('Gridding mesh file in %s \n' % WL.wd[i])
            _, _, etagrid = WL.mesh_to_grid(mesh_subs, dx=args.dxy, dy=args.dxy, 
                    xlim=(args.xmin, args.xmax+1), ylim=(args.ymin, args.ymax+1),
                    vertices=vertices, weights=weights)
            # Save eta grid to netcdf
            ds.Z[i,:,:] = etagrid
            # ds.Z.iloc[dict(time=i)] = etagrid
            
            # Also save compressed cam0 raw image in JPG format
            fn_cam0 = os.path.join(wd, '00000000_s.png')
            im0 = cv2.imread(fn_cam0, cv2.IMREAD_GRAYSCALE)
            # Compress to jpg
            # ret, imgjpeg = cv2.imencode('.jpg', im0)
            ds.cam0images[i,:,:] = im0
            # ds.cam0images.iloc[dict(time=i)] = imgjpeg

#     # wassncplot 'meta' attributes
#     ds.meta['p0plane'] = WL.plane
#     ds.meta['image_width'] = args.iw
#     ds.meta['image_height'] = args.ih
#     ds.meta['zmin'] = ds.Z.min().item()
#     ds.meta['zmax'] = ds.Z.max().item()
#     ds.meta['zmean'] = 0

    # Units & attributes
    ds.time.encoding['units'] = time_units
    ds.time.attrs['units'] = time_units
    ds.time.attrs['standard_name'] = 'time'
    ds.time.attrs['long_name'] = 'Right camera frame timestamp (UTC)'
    # Reconstructed sea surface
    ds.Z.attrs['units'] = 'm'
    ds.Z.attrs['standard_name'] = 'sea_surface_height_above_mean_sea_level'
    ds.Z.attrs['long_name'] = 'WASS-reconstructed sea surface grid'
    # x,y 
    ds.X.attrs['units'] = 'm'
    ds.X.attrs['standard_name'] = 'projection_x_coordinate'
    ds.X.attrs['long_name'] = 'x coordinate in local camera coordinate system'
    ds.Y.attrs['units'] = 'm'
    ds.Y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.Y.attrs['long_name'] = 'y coordinate in local camera coordinate system'
    # x,y grids
    ds.X_grid.attrs['units'] = 'm'
    ds.X_grid.attrs['standard_name'] = 'projection_x_coordinate'
    ds.X_grid.attrs['long_name'] = 'x grid in local camera coordinate system'
    ds.Y_grid.attrs['units'] = 'm'
    ds.Y_grid.attrs['standard_name'] = 'projection_y_coordinate'
    ds.Y_grid.attrs['long_name'] = 'y grid in local camera coordinate system'
    # Globl attributes
    ds.attrs['summary'] =  ("Stereo video from ROXSI II experiment at Hopkins " + 
                            "Marine Station in Monterey, CA.")
    ds.attrs['cam0 lens serial number'] =  args.ser_cam0
    ds.attrs['cam1 lens serial number'] =  args.ser_cam1
    ds.attrs['image_width'] = args.iw
    ds.attrs['image_height'] = args.ih
    ds.attrs['zmin'] = ds.Z.min().item()
    ds.attrs['zmax'] = ds.Z.max().item()
    ds.attrs['zmean'] = 0

    # Set netcdf encoding before saving
    encoding = {'time': {'zlib': False, '_FillValue': None},
                'X': {'zlib': False, '_FillValue': None},
                'Y': {'zlib': False, '_FillValue': None},
                'V4': {'zlib': False, '_FillValue': None},
                'ih': {'zlib': False, '_FillValue': None},
                'iw': {'zlib': False, '_FillValue': None},
                }     
    # Set variable fill values
    for k in list(ds.keys()):
        encoding[k] = {'_FillValue': args.fillvalue}

    # Save netcdf
    ds.to_netcdf(fn_nc, encoding=encoding)

# Read netcdf
ds = xr.decode_cf(xr.open_dataset(fn_nc, decode_coords='all'))

print(' ')
print('Done.')

