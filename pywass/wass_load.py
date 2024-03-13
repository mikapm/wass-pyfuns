#!/usr/bin/env python3
"""
Load WASS output meshes and prepare for analysis.
"""
import numpy as np
import cv2
import array
import os
import glob
from datetime import datetime as DT
from scipy.interpolate import griddata

def fread(fid, nelements, dtype, out_shape=None):
    """
    Hack to emulate Matlab's fread() function
    for reading binary files.

    Parameters:
        fid - file identifier object
              ex. fid=open('filename')
        nelements - int; number of elements
                    to read
        dtype - numpy dtype object
        out_shape - tuple; shape other than 1D 
                    array for returned data_array
    
    Borrowed from:
    https://stackoverflow.com/questions/34026326/
    ensuring-python-equivalence-of-matlabs-fread/
    34027466#34027466

    Example:
        fid = open('mesh_cam.xyzC', 'rb')
        print(fread(fid, 1, np.uint32))
    
    """
    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    # Read requested contents
    data_array = np.fromfile(fid, dt, nelements)
    if out_shape is None:
        data_array.shape = (nelements, 1)
    else:
        data_array.shape = out_shape
    # Squeeze if need be
    data_array = data_array.squeeze()

    return data_array

def interpolate(values, vtx, wts, fill_value=np.nan):
    """
    Interpolate points to grid using vertices and weights from
    interp_weights(). See that function for example. Generates output
    that is identical to scipy.interpolate.griddata.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def read_PtGrey_timestamp(im):
    """
    Returns the embedded timestamp from a non-post-processed
    PointGrey image frame.
    Based on Matlab function readPtGreyTimeStamp.m by Michael
    Schwendeman. Addidtional information found at 
    https://groups.google.com/forum/#!topic/bonsai-users/WD6mV94KAQs.
    Parameters:
        im - uint8 ndarray; PointGrey image array.
    #    filename - str; if not None, will get yyyy-mm-dd HH:MM:SS
    #               format time from image filename. Note: must be
    #               the filename of the input image, and must follow
    #               the PointGrey image naming convention as used at
    #               APL, e.g. flea83_2019-12-10-181702-0000.pgm
    Returns:
        timestamp - float; PointGrey timestamp in seconds
    """


    # The timestamp information is in the first 32 pixels in the
    # upper left corner of the image
    embedded_pixels = im[0,0:32]
    # Convert pixels to binary strings (same format as Matlab)
    bin_strings = [format(i,'08b') for i in embedded_pixels]
    # Join strings in bin_strings into one long binary string
    embedded_string = ''.join(bin_strings)
    # Parse the embedded string
    # The first 7 bits is the second count
    sec_count = int(embedded_string[:7], 2)
    # The following 13 bits are the cycles
    cycle_count = int(embedded_string[7:20], 2)
    # The following 12 bits give the offset
    cycle_offset = int(embedded_string[20:32], 2)
    # Calculate the timestamp (not entirely sure what's going on
    # here)
    timestamp = (sec_count + (cycle_count + cycle_offset/3072) / 8000)

    return timestamp


def uncycle_PtGrey_timestamps(time):
    """
    From
    https://groups.google.com/forum/#!topic/bonsai-users/WD6mV94KAQs:
        'Now, we still have the problem that this 
        representation will cycle every 128 seconds. 
        To uncycle it, we need to find where it's cycling.
        Fortunately, time only moves forward :-) so we can
        get where the cycles are by looking for the points
        where the difference in time is negative. 
        Then you only need to accumulate those cycles and 
        shift the cyclic time accordingly.'

    Since Ekofisk stereo video images taken prior to February 2020
    had a bug in the naming of images which caused some frames to
    be one second 'off', this function requires the difference
    between two timestamps to be larger than 100. This way the
    cycling ignores the shorter 1-sec offsets that may be found in
    Ekofisk timestamps.
    """
    cycles = np.insert(np.diff(time) < -100, 0, False)
    cycleindex = np.cumsum(cycles)

    return time + cycleindex * 128

class WASS_load():
    """
    Define scale (i.e. baseline in m) and set working directory
    in self.data_root.

    Example:
        import wass_load as wl
        WL = wl.WASS_load(data_root='/home/mikapm/wass/ekok/exp/exp_20190317/')
        mesh = WL.align_plane() 

    """
    def __init__(self, 
            data_root = '/home/mikapm/wass/ekok/exp/exp_20191209',
            scale = 5.11, # scaling multiple for point cloud (=baseline)
            incorrect_ordering = False,
            plane = 'sub', # which average plane to use to align mesh
            ):
        self.data_root = data_root # Root working directory
        self.scale = scale # baseline (in metres)
        self.incorrect_ordering = incorrect_ordering
        # Plane type
        if plane in ['sub', 'pipe', 'all']:
            self.plane = plane
        else:
            raise ValueError('Set valid plane arg.')
        # List all working directories
        self.wd = sorted(glob.glob(os.path.join(self.data_root,
            'out', '*wd')))
        # Mean sea plane dir.
        self.planedir = os.path.join(self.data_root, 'planes')
        # Also list input images (which provide timestamps)
        self.in_cam0 = sorted(glob.glob(os.path.join(
            self.data_root, 'input', 'cam0', '*.tif')))
        self.in_cam1 = sorted(glob.glob(os.path.join(
            self.data_root, 'input', 'cam1', '*.tif')))
        # Get the correct image order (only required for pre-Feb
        # 2020 images)
        if self.incorrect_ordering:
            self.corr_seq_file = os.path.join(self.data_root, 
                    'corr_seq.txt')
            if not os.path.exists(self.corr_seq_file):
                self.correct_sequence = self.find_correct_sequence()
                # Save to file
                np.savetxt(self.corr_seq_file, self.correct_sequence,
                        fmt='%i')
            else:
                self.correct_sequence = np.array(
                        np.loadtxt(self.corr_seq_file), dtype=int)
        # Grid directory
        self.grid_dir = os.path.join(self.data_root, 'grid')
        # Date and experiment start time from data root dir name
        # Assuming data_root directory structure:
        # .../date[yyyymmdd]/starttime[HHMM]
        self.date = os.path.normpath(self.data_root).split('/')[-2]
        self.starttime = os.path.basename(os.path.normpath(
            self.data_root))
        

    def find_correct_sequence(self):
        """
        Prior to Feb 2020 the Ekofisk stereo images had an issue
        with some of the timestamps in the image filenames being
        incorrect. This happened because the microseconds in the
        timestamps were taken from the cameras, while the hours and
        minutes were obtained from the stereo camera PC, and the
        cameras and PC are not perfectly synchronized. Therefore
        some frames (usually the first/last in a one-second
        interval) have the wrong timestamp. This is apparent when
        plotting sequential frames, as some frames seem to jump
        back and forth.
        
        The solution to this is to use the embedded pixel
        timestamps that are natively stored on all PointGrey
        images.

        This function returns a list of the correct sequence
        indexing to be applied on a sorted list of incorrectly
        sequenced images. Example usage: 

            np.array(self.in_cam0)[corr_seq]

        """
        # Loop over all input images of cam0 and get timestamps
        timestamps = np.zeros(len(self.in_cam0))
        print('Finding correct image sequence ... \n')
        for i,fn in enumerate(self.in_cam0):
            if i % 100 == 0:
                print('%s \n' % i)
            im=cv2.imread(fn, 0)
            timestamps[i] = read_PtGrey_timestamp(im)
        # Correct for the 128-sec cycling in the timestamps
        timestamps = uncycle_PtGrey_timestamps(timestamps)
        # Sort the cycled timestamps to get the correct image order
        # (in the for of list indices)
        corr_seq = np.argsort(timestamps)

        return corr_seq


    def load_camera_mesh(self, idx, print_msg=False):
        """
        Load the stereo mesh created by WASS.

        Parameters:
            idx - int; working directory number starting
                  at 0.
            print_msg - Set to False to ignore any print statements
        Returns:
            mesh_cam; ndarray - 2D mesh created by WASS

        Example for visualizing mesh:
            import plotly.graph_objects as go
            X,Y,Z = mesh.T
            fig = go.Figure(data=[go.Mesh3d(x=X, y=Y, z=Z, opacity=0.50)])
            fig.show(renderer="browser")

        TODO: Option to read uncompressed mesh
        """
        mesh_dir = self.wd[idx]
        mesh_file = os.path.join(mesh_dir, 'mesh_cam.xyzC')
        if print_msg:
            print('Mesh file: %s \n' % mesh_file)
        with open(mesh_file, 'rb') as fid:
            if print_msg:
                print('Loading compressed data ...\n')
            # Use fread defined above to read file
            # contents
            npts = fread(fid, 1, np.uint32)
            # xmax, ymax, zmax, xmin, ymin, zmin (?)
            # used in quantization of axes
            limits = fread(fid, 6, np.float)
            # Read rotation matrix inverse
            Rinv = fread(fid, 9, np.float, (3,3))
            #Rinv = Rinv.T
            # Read translation matrix inverse
            Tinv = fread(fid, 3, np.float)
            # Rest of file is the mesh
            mesh_cam = fread(fid, 3*npts, np.uint16,
                    (npts,3))
            # Turn into float and transpose
            mesh_cam = np.array(mesh_cam, dtype=np.float).T
            # Scale mesh
            lims_max = limits[:3].reshape(3,1) # max. x,y,z limits
            lims_min = limits[3:].reshape(3,1) # min. x,y,z limits
            mesh_cam = (mesh_cam / 
                    np.tile(lims_max,(1,mesh_cam.shape[1])) + 
                    np.tile(lims_min,(1,mesh_cam.shape[1]))
                    )
            # Rotate and translate
            mesh_cam = (Rinv @ mesh_cam + 
                    np.tile(Tinv.reshape(3,1), (1,mesh_cam.shape[1]))
                    )

        return mesh_cam


    def RT_from_plane(self, plane):
        """ 
        Compute rotation and translation matrices for mean sea plane
        estimate from WASS plane file.

        Parameters:
            plane - ndarray; WASS plane array

        Returns:
            R - 3x3 rotation matrix
            T - 3x1 translation vector
        """
        # Compute R and T from plane
        a, b, c, d = plane
        q = (1-c) / (a*a + b*b)
        R = np.eye(3)
        T = np.zeros(3).reshape(-1,1)
        R[0,0] = 1 - a*a*q
        R[0,1] = -a*b*q
        R[0,2] = -a
        R[1,0] = -a*b*q
        R[1,1] = 1 - b*b*q
        R[1,2] = -b
        R[2,0] = a
        R[2,1] = b
        R[2,2] = c
        T[2] = d

        return R, T


    def align_plane(self, idx=1, mesh=None, plane=None, planefile=None, 
            return_plane=False, print_msg=False):
        """
        Align generated mesh by rotation and translation
        using R and T matrices following the Matlab function
        load_camera_mesh_and_align_plane() by Filippo Bergamasco.

        Parameters:
            idx - int; as idx in self.load_camera_mesh()
            mesh - ndarray; camera mesh returned by
                   self.load_camera_mesh(). If None, that
                   function is first called to load the mesh
                   specified by idx.
            plane - ndarray; WASS plane array
            planefile - str; path to plane file
            print_msg - Set to False to ignore any print statements
        """
        if mesh is None:
            mesh = self.load_camera_mesh(idx)
        if plane is None:
            # Load averaged plane if planefile not specified
            if planefile is None:
                # Use average plane (if generated)
                try:
                    planefile = os.path.join(self.data_root,
                            'plane_avg_{}.txt'.format(self.plane))
                    plane = np.loadtxt(planefile)
                except FileNotFoundError:
                    raise ValueError('Generate plane_avg_{}.txt file '
                        'or specify planefile'.format(self.plane))
            else:
                plane = np.loadtxt(planefile)

        # Compute R and T from plane
        R, T = self.RT_from_plane(plane)

        # Rotate and translate and transpose
        if print_msg:
            print('Aligning mesh according to plane ... \n')
        mesh_aligned = (R @ mesh + np.tile(T, (1, mesh.shape[1]))).T
        # Apply scale, i.e. multiply by baseline to get units in metres
        mesh_aligned = mesh_aligned * self.scale
        # Invert z axis
        mesh_aligned[:,2] *= (-1)

        if return_plane:
            return mesh_aligned, plane
        else:
            return mesh_aligned


    def mesh_to_grid(self, mesh_aligned, dx, dy, print_msg=False,
            xlim=None, ylim=None, vertices=None, weights=None,
            **kwargs):
        """
        Interpolate WASS point cloud mesh into structured (x,y) grid. 
        The grid's resolution is defined by dx and dy. 
        xlim and ylim should be tuples, e.g. (xmin, ymin), or if not 
        specified, max and min values of mesh X and Y values will be used.
        Uses scipy.interpolate.griddata() function.

        To speed up interpolation of several variables from the same mesh,
        e.g. pixel coordinates, compute vertices and weights with
        sw_pyfuns.gridding.interp_weights() and insert as 'vertices' and
        'weights'. This way the (slow) Delaunay triangulation is only performed
        once, i.e. when interp_weights() is called.

        **kwargs as for griddata(), e.g. method='linear'.

        Returns gridded data as xgrid, ygrid, zgrid
        """
        # Separate mesh coordinates
        if len(mesh_aligned.T.shape) == 2:
            X, Y, Z = mesh_aligned.T
        elif len(mesh_aligned.T.shape) == 1:
            Z = mesh_aligned.T
        if xlim is None:
            xlim = (np.min(X), np.max(X))
        if ylim is None:
            ylim = (np.min(Y), np.max(Y))
        # Make output x,y grid
        xgrid, ygrid = np.mgrid[xlim[0]:xlim[1]:dx, ylim[0]:ylim[1]:dy]
        # Interpolate
        if vertices is not None:
            if print_msg:
                print('Interpolating mesh to regular grid with speed-up ... \n')
            ny, nx = xgrid.shape
            zgrid = interpolate(Z, vertices, weights)
            zgrid = zgrid.reshape(ny, nx)
        else:
            if print_msg:
                print('Interpolating mesh to regular grid ... \n')
            zgrid = griddata((X,Y), Z, (xgrid,ygrid), **kwargs)

        return xgrid, ygrid, zgrid


    def nan_grid(self, mesh_aligned, xgrid, ygrid):
        """
        Make x,y,z grid where z=NaN in points with no values in
        mesh_aligned. Can be used to mask out interpolated grid
        points in zgrid from self.mesh_to_grid().
        Returns mask grid with dimensions (xgrid,ygrid) where
        NaN points are assigned True.
        """

#        def bindata(x, dx=-10, x0=None, x1=None):
#            """Bin vector x according to resolution dx
#
#               ix = bindata(x, dx, x0, x1)
#
#            Returns the vector of indices, starting from 0, corresponding to the chosen 
#            bin size, dx, start x0 and end x1. If x1 is omitted, x1 = max(x) - dx/2. 
#            If x0 is omitted, x0 = min(x) + dx/2. If dx is omitted, the data 
#            are divided into 10 classes. Note that outliers are not removed. When dx is
#            negative it is interpreted as the number of classes.
#
#            Example
#            -------
#
#            >>> i = bindata(hs,1,0.5)
#          
#            Bin all elements in hs in bins of width 1, first bin centered on 0.5.
#         
#            17.1.97, Oyvind.Breivik@gfi.uib.no.
#            2007-06-25, Oyvind.Breivik@met.no. Added vector binning.
#            2012-06-12, Converted to Python 2.7, oyvind.breivik@ecmwf.int
#            """
#          
#            if dx < 0:
#                Dx = (x1-x0)/dx
#            else:
#                Dx = dx
#            if x0 is None:
#                X0 = np.min(x) + Dx/2
#            else:
#                X0 = x0
#            if x1 is None:
#                X1 = np.max(x) - Dx/2
#
#            #return np.round((x - X0)/Dx)
#            return np.floor((x - X0)/Dx)

        # Separate mesh coordinates
        if len(mesh_aligned.T.shape) == 2:
            X, Y, Z = mesh_aligned.T
            Y *= -1
        elif len(mesh_aligned.T.shape) == 1:
            Z = mesh_aligned.T
#        # Make list with mesh points to check
#        points = list(map(list, zip(X, Y)))
#        mpath = Path(points)
#        # Flatten x,y grids
#        coord = np.dstack((xgrid, ygrid)).reshape(-1,2)
#        within = coord[mpath.contains_points(coord)]
#        h,w = X.shape
#        mask = np.zeros((h,w))
#        mask[within[:,1], within[:,0]] = 1

        X0 = np.nanmin(xgrid)
        X1 = np.nanmax(xgrid)
        Y0 = np.nanmin(ygrid)
        Y1 = np.nanmax(ygrid)

        Dx = abs(xgrid[1,0] - xgrid[0, 0])
        Dy = abs(ygrid[0,1] - ygrid[0, 0])
        nx = int(round((X1 - X0) / Dx) + 1)
        #X1 = X0 + (nx-1) * Dx
        #X1 = np.min((X0 + (nx-1) * Dx, nx-1))
        ny = int(round((Y1 - Y0) / Dy) + 1)
        #Y1 = np.min((Y0 + (ny-1) * Dy, ny-1))

        # Bin data in x,y-space
        #ixo = bindata(X, Dx, X0, X1)
        ix = np.digitize(X, np.arange(X0, X1, Dx))
        #iyo = bindata(Y, Dy, Y0, Y1)
        iy = np.digitize(Y, np.arange(Y0, Y1, Dy))

#        # Throw away outliers
#        ingrid = (ix >= 0) & (ix <= nx) & (iy >= 0) & (iy <= ny)
#        ix = np.array(ix[ingrid]).astype(int)
#        iy = np.array(iy[ingrid]).astype(int)
#        X = X[ingrid]
#        Y = Y[ingrid]
#        Z = Z[ingrid]
        N = len(ix) # how many data points are left now?

        # Place data points in cells
        ngrid = np.zeros([nx, ny]) # no of obs per grid cell
        for i in range(N):
            #ngrid[ix[i], iy[i]] = ngrid[ix[i], iy[i]] + 1
            ngrid[ix[i], iy[i]] += 1

        # Return mask for grid points with zero observations
        Nil = (ngrid==0)

        return Nil

        
    def project_mesh_to_cam(self, idx=1, plane=None, mesh=None, cam=1,
                            print_msg=False):
        """
        Project mesh onto undistorted camera frame view.
        Specify camera number 0 (left) or 1 (right).
        plane should be an ndarray, e.g. the plane returned
        by self.align_plane().
        """
        if mesh is None:
            mesh = self.load_camera_mesh(idx, print_msg=print_msg)
        # Load projection matrix
        P1 = np.loadtxt(os.path.join(self.wd[idx], 'P{}cam.txt'.format(cam)))
        # Get elevations from mesh and plane
        n_pts = mesh.shape[1]
        # Matlab dot(a,b) -> np.sum(a.conj() * b, axis=0)
        elevations = np.sum(np.vstack((mesh, np.ones(n_pts))).conj() * 
                np.tile(plane.reshape(-1,1), (1,n_pts)), axis=0)
        elevations *= (self.scale * (-1)) # fix scale and z direction
        # Project mesh onto camera view
        if print_msg:
            print('Projecting unaligned mesh to camera view ... \n')
        pt2d = P1 @ np.vstack((mesh, np.ones(n_pts)))
        # Homogeneous -> Cartesian coord: divide by last coordinate
        pt2d /= np.tile(pt2d[2,:], (3,1))

        return elevations, pt2d


    def timestamp_from_fname(self, idx):
        """
        Get timestamp from input image filename. Uses cam0 filenames
        by default. Returns timestamp in datetime.datetime object
        format.
        """
        # Get date string
        #datestr = os.path.basename(os.path.split(self.data_root)[0])
        datestr = self.date
        # Get image filename (use cam0)
        fn = os.path.basename(self.in_cam0[idx])
        # Get time string
        timestr = fn.split('_')[1]
        # Combine date and time strings and make DT object
        timestamp = DT.strptime(datestr + ' ' + timestr, '%Y%m%d 0%H%M%S%f')

        return timestamp





