#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
from datetime import datetime as DT
from datetime import timedelta as TD
from sw_pyfuns.wass_stuff import wass_load as wlo
from sw_pyfuns import image_processing as swi
from sw_pyfuns import gridding as swg
from argparse import ArgumentParser

"""
Script that generates x,y grids of all meshes in 
a WASS experiment directory. The grids are all 
saved in one netCDF file in the root experiment
directory.
"""

def parse_args(**kwargs):
    """
    Input arguments for main script
    """
    parser = ArgumentParser()

    parser.add_argument("-dr", 
            help=("WASS experiment root directory."),
            type=str,
            default='/lustre/storeB/project/fou/om/stereowave/data/wass_output/20200104/0940',
            )
    parser.add_argument("-dxy", 
            help=("Grid resolution (in x and y) in meters."),
            type=float,
            default=0.5,
            )
    parser.add_argument("-xmin", 
            help=("Lower grid x limit."),
            type=float,
            default=-70,
            )
    parser.add_argument("-xmax", 
            help=("Upper grid x limit."),
            type=float,
            default=70,
            )
    parser.add_argument("-ymin", 
            help=("Lower grid y limit. (y decreases outwards from platform)"),
            type=float,
            default=-200,
            )
    parser.add_argument("-ymax", 
            help=("Upper grid y limit."),
            type=float,
            default=-50,
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
    parser.add_argument("--corr_seq", 
            help=("Correct grid sequence ordering (EKOK pre-Feb 2020)?"),
            action="store_true",
            )

    return parser.parse_args(**kwargs)

args = parse_args(args=sys.argv[1:])

# Initialize WASS_load object
if args.corr_seq:
    WL = wlo.WASS_load(data_root=args.dr, incorrect_ordering=True,
            plane=args.plane)
    corr_seq = WL.correct_sequence.copy()
else:
    WL = wlo.WASS_load(data_root=args.dr, plane=args.plane)
    corr_seq = np.arange(len(WL.wd))

# Make grid directory if it doesn't yet exist
if not os.path.exists(WL.grid_dir):
    print('Making grid directory ... \n')
    os.mkdir(WL.grid_dir)

# Get first timestamp
ts0 = WL.timestamp_from_fname(0)
# Units for netcdf timestamp
unout = 'microseconds since {}'.format(DT.strftime(ts0, '%Y-%m-%d %H:%M:%S:%f'))
# netCDF file names
nc_name_xy = os.path.join(WL.grid_dir, 'xygrid_{}cm_{}_{}_plane_{}.nc'.format(
    int(args.dxy*100), WL.date, WL.starttime, args.plane))
nc_name_im = os.path.join(WL.grid_dir, 'imgrid_{}cm_{}_{}_plane_{}.nc'.format(
    int(args.dxy*100), WL.date, WL.starttime, args.plane))

# Initialize output netCDF file
nc_out_xy = nc.Dataset(nc_name_xy, 'w')
nc_out_im = nc.Dataset(nc_name_im, 'w')

# Make time nc variable
nc_out_xy.createDimension('time', len(corr_seq));
nc_out_im.createDimension('time', len(corr_seq));
timevar_xy = nc_out_xy.createVariable('time', 'float64', ('time'))
timevar_xy.setncattr('units', unout)
timevar_im = nc_out_im.createVariable('time', 'float64', ('time'))
timevar_im.setncattr('units', unout)

# Initialize common grids
xgrid, ygrid = np.mgrid[args.xmin:args.xmax:args.dxy, 
    args.ymin:args.ymax:args.dxy]
# xy_grid needed for comuting vertices and weights for interpolation
xy_grid = np.vstack((xgrid.flatten(),ygrid.flatten())).T

# Define nc dimensions
nx, ny = xgrid.shape
xvec = xgrid[:,0]
yvec = ygrid[0,:]
nc_out_xy.createDimension('x',nx);
nc_out_xy.createDimension('y',ny);
nc_out_im.createDimension('x',nx);
nc_out_im.createDimension('y',ny);

# Create nc variables
print('Creating nc variables ... \n')
x_var_xy = nc_out_xy.createVariable('x','float32',('x'))
x_var_xy.setncattr('units','m') 
x_var_xy[:] = xvec
x_var_im = nc_out_im.createVariable('x','float32',('x'))
x_var_im.setncattr('units','m') 
x_var_im[:] = xvec

y_var_xy = nc_out_xy.createVariable('y','float32',('y'))
y_var_xy.setncattr('units','m') 
y_var_xy[:] = yvec
y_var_im = nc_out_im.createVariable('y','float32',('y'))
y_var_im.setncattr('units','m') 
y_var_im[:] = yvec

xgrid_var_xy = nc_out_xy.createVariable('xgrid','float32', ('x','y'))
xgrid_var_xy[:] = xgrid
ygrid_var_xy = nc_out_xy.createVariable('ygrid','float32', ('x','y'))
ygrid_var_xy[:] = ygrid

xgrid_var_im = nc_out_im.createVariable('xgrid','float32', ('x','y'))
xgrid_var_im[:] = xgrid
ygrid_var_im = nc_out_im.createVariable('ygrid','float32', ('x','y'))
ygrid_var_im[:] = ygrid

etagrid_var = nc_out_xy.createVariable('eta','float32', ('time','x','y'))
imgrid_var = nc_out_im.createVariable('im_grid','uint8', ('time','x','y'))

# Take difference of sequencing indices to detect when sequencing is off when
# constructing timestamps
seq_diff = np.diff(corr_seq)
# Loop over work directories and perform gridding
for cnt, i in enumerate(tqdm(corr_seq)):
    if os.path.isfile(os.path.join(WL.wd[i], 'mesh_cam.xyzC')):
        # Load mesh
        mesh = WL.load_camera_mesh(idx=i)
        # Align with plane
        mesh_al, plane = WL.align_plane(mesh=mesh, return_plane=True)
        # Project mesh to cam1 view for pixel coordinates
        _, pt2d = WL.project_mesh_to_cam(i, plane, mesh)
        pt2d = pt2d.T # Transpose to same shape as mesh_al

        # Exclude points outside xmin, xmax & ymin, ymax
        x,y,z = mesh_al.T
        mesh_al = mesh_al[(x>args.xmin) & (x<args.xmax) & \
                (y>args.ymin) & (y<args.ymax), :]
        pt2d = pt2d[(x>args.xmin) & (x<args.xmax) & (y>args.ymin) & (y<args.ymax),:]
        # x,y,z coordinates from (reduced) aligned mesh
        x,y,z = mesh_al.T

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
        x,y,z = mesh_subs.T
        xy = np.vstack((x.flatten(),y.flatten())).T 
        print('Computing vertices \n')
        vertices, weights = swg.interp_weights(xy, xy_grid)

        # Grid mesh to eta grid and pixel coordinates to iR and jR
        print('Gridding mesh file in %s \n' % WL.wd[i])
        _, _, etagrid = WL.mesh_to_grid(mesh_subs, dx=args.dxy, dy=args.dxy, 
                xlim=(args.xmin, args.xmax), ylim=(args.ymin, args.ymax),
                vertices=vertices, weights=weights)
        # Save eta grid to netcdf
        etagrid_var[cnt, :,:] = etagrid

        # u pixel coordinates
        print('Gridding u pixel coordinates \n')
        _, _, iR = WL.mesh_to_grid(pt2d_subs[:,0], dx=args.dxy, dy=args.dxy, 
                xlim=(args.xmin, args.xmax), ylim=(args.ymin, args.ymax),
                vertices=vertices, weights=weights)
        iR = np.array(iR).astype(np.float32)

        # v pixel coordinates
        print('Gridding v pixel coordinates \n')
        _, _, jR = WL.mesh_to_grid(pt2d_subs[:,1], dx=args.dxy, dy=args.dxy, 
                xlim=(args.xmin, args.xmax), ylim=(args.ymin, args.ymax),
                vertices=vertices, weights=weights)
        jR = np.array(jR).astype(np.float32)

        # Load undistorted right camera image
        imfn = os.path.join(WL.wd[i], 'undistorted', '00000001.png')
        im = cv2.imread(imfn, 0)
        # Remap input image to aligned mesh projection using iR and jR
        imgrid = cv2.remap(im, iR, jR, cv2.INTER_LINEAR)
        # Save im grid to netcdf
        imgrid_var[cnt, :,:] = imgrid

        # Get timestamp for current grid
        timestamp = WL.timestamp_from_fname(i)
        # Is timestamp correct?
        if cnt > 0 and seq_diff[cnt-1] < 0:
            # If diff is <0 => incorrect input image filename
            # The fix is to add one second to the timestamp
            print('Adding a second to timestamp ... \n')
            timestamp += TD(seconds=1)
        timestamp_nc = nc.date2num(timestamp, unout)
        timevar_xy[cnt] = timestamp_nc
        timevar_im[cnt] = timestamp_nc


# Close netcdf file
nc_out_xy.close()
nc_out_im.close()



