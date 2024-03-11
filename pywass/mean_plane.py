#!/usr/bin/env python3
"""
Estimate mean sea plane over a select region of WASS point clouds.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import wass_load as wlo


def compute_similarity(mesh):
    """
    Compute similarity transformation to normalize data to zero mean and
    mean distance to the origin equal to 1 per coordinate.

    Based on ComputeSimilarity.m by 
    Filippo Bergamasco / Alvise Benetazzo.

    Parameters:
        mesh - ndarray; WASS point cloud

    Returns:
        mesh_sim - ndarray; similarity transformed point cloud
        H - ndarray; transformation matrix
    """

    # Get shape
    n_points, dim = mesh.shape
    if dim > n_points:
        # Transpose mesh for correct shape
        mesh = mesh.T
        n_points, dim = mesh.shape

    # Compute translation and centred points
    print('Computing similarity ... \n')
    t = np.mean(mesh, 0) # translation vector (centroid)
    mesh_sim = mesh - np.tile(t, (n_points,1))

    # Compute scaling and scaled points from centred points
    mean_distance = np.mean(np.sqrt(np.sum(mesh_sim * np.conj(mesh_sim), 1)))
    scale = np.sqrt(dim) / mean_distance # scaling factor
    mesh_sim *= scale

    # Transformation matrix & transformed points in homogeneous coords
    H = np.eye(dim + 1)
    H[:-1,-1] = -t.T # Translation
    H *= scale # Scaling
    H[-1, -1] = 1 # last coord in homog. coord should be 1

    t = -t.T

    return mesh_sim, H



def fit_plane_simple(mesh):
    """
    Given a set of 3D points, fit a plane the compute the rotation
    and translation that converts it into the plane z=0.

    Based on mean_plane_simplest_method.m by 
    Filippo Bergamasco / Alvise Benetazzo.
    """
    # 1. Normalize data by a similarity transformation to improve numerical
    #    conditioning
    mesh_sim, H = compute_similarity(mesh)

    # 2. Fit a plane by the simplest algebraic method
    print('Fitting plane by svd ... \n')
    u, d, v = np.linalg.svd(mesh_sim, full_matrices=False)
    plane = np.zeros(len(v) + 1)
    plane[:3] = v[2,:]

    # 3. Undo the effect of the similarity transformation on the coordinates
    #    of the plane.
    plane = H.T @ plane

    # 4. Make the plane have unit normal. 
    #    Normalize plane so that (a^2+b^2+c^2)=1
    plane = plane / np.linalg.norm(plane[:3])
    if plane[2] < 0:
        # If dot product of plane(1:3) and [0;0;1] is negative, then use the
        # other normal of the plane.
        plane *= -1
    a = plane[0]
    b = plane[1]
    c = plane[2]

    # 5. Compute rotation matrix when (a^2+b^2+c^2)=1
    R = np.zeros((3,3))
    R[0,0] = 1 - (1 - c) * (a**2) / (a**2 + b**2)
    R[0,1] = -(1 - c) * a * b / (a**2 + b**2)
    R[0,2] = -a
    R[1,0] = -(1 - c) * a * b / (a**2 + b**2)
    R[1,1] = 1 - (1 - c) * (b**2) / (a**2 + b**2)
    R[1,2] = -b
    R[2,:] = a, b, c

    # 6. Compute Translation vector
    T = np.array([0, 0, plane[3]]);

    return plane, R, T


if __name__ == '__main__':

    from argparse import ArgumentParser

    # Input args
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
        parser.add_argument("-baseline", 
                help=("Stereo baseline in m."),
                type=float,
                default=5.11,
                )
        parser.add_argument("-xmin", 
                help=("Lower x limit for plane estimation region."),
                type=float,
                default=-70,
                )
        parser.add_argument("-xmax", 
                help=("Upper x limit for plane estimation region."),
                type=float,
                default=70,
                )
        parser.add_argument("-ymin", 
                help=("Lower plane y limit. (y decreases outwards from platform)"),
                type=float,
                default=-120,
                )
        parser.add_argument("-ymax", 
                help=("Upper y limit for plane estimation region."),
                type=float,
                default=-50,
                )
        parser.add_argument("-ind_s", 
                help=("Start index."),
                type=int,
                default=0,
                )
        parser.add_argument("-ind_e", 
                help=("End index."),
                type=int,
                default=-1,
                )

        return parser.parse_args(**kwargs)

    args = parse_args(args=sys.argv[1:])
    


    # Make WASS_load object
    WL = wlo.WASS_load(data_root=args.dr, scale=args.baseline)

    # Store all planes for averaging
    planes_pipe = np.zeros((len(WL.wd), 4)).astype(np.float32) # from WASS pipeline
    planes_all = np.zeros_like(planes_pipe) # from entire mesh region
    planes_sub = np.zeros_like(planes_pipe) # from sub-region of mesh

    # Loop over working directories and compute planes
    for i, wd in enumerate(tqdm(WL.wd)):
        print(' ')

        # Load pipeline-generated plane
        fn_plane_pipe = os.path.join(wd, 'plane.txt')
        if os.path.isfile(fn_plane_pipe):
            with open(fn_plane_pipe, 'r') as fn:
                planes_pipe[i,:] = np.loadtxt(fn)
        else:
            print('Plane file not found in %s \n' % wd)

        fn_mesh = os.path.join(wd, 'mesh_cam.xyzC')
        if os.path.isfile(fn_mesh):

            # Load mesh
            mesh_cam = WL.load_camera_mesh(i)

            # Mean plane all points
            plane_all, _, _ = fit_plane_simple(mesh_cam)
            # Align plane
            mesh_al = WL.align_plane(mesh=mesh_cam, plane=plane_all)

            # Mean plane sub region
            x,y,z = mesh_al.T
            mesh_cam = mesh_cam.T
            mesh_sub = mesh_cam[(x>args.xmin) & (x<args.xmax) & \
                    (y>args.ymin) & (y<args.ymax), :]
            plane_sub, _, _ = fit_plane_simple(mesh_sub)

            # Save planes
            planes_all[i,:] = plane_all
            planes_sub[i,:] = plane_sub

    # Save all planes
    print(' ')
    print('Saving planes \n')
    np.savetxt(os.path.join(WL.data_root, 'planes_pipe.txt'), planes_pipe)
    np.savetxt(os.path.join(WL.data_root, 'planes_all.txt'), planes_all)
    np.savetxt(os.path.join(WL.data_root, 'planes_sub.txt'), planes_sub)

    # Average planes and save average planes
    print('Averaging planes and saving \n')
    plane_avg_pipe = np.mean(planes_pipe, 0)
    plane_avg_all = np.mean(planes_all, 0)
    plane_avg_sub = np.mean(planes_sub, 0)
    np.savetxt(os.path.join(WL.data_root, 'plane_avg_pipe.txt'), plane_avg_pipe)
    np.savetxt(os.path.join(WL.data_root, 'plane_avg_all.txt'), plane_avg_all)
    np.savetxt(os.path.join(WL.data_root, 'plane_avg_sub.txt'), plane_avg_sub)




