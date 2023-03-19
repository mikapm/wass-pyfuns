#!/usr/bin/env python3
""" 
Functions for running the WASS [1] pipeline (in parallel)
from a terminal, i.e. without the web-based interface.

Based on the Matlab script ab_launch_pipeline_kiost.m
by Alvise Benetazzo and Filippo Bergamasco of ISMAR, Venice.

Parallelisation based on the subprocess library
following the example in 
https://gist.github.com/davidgardenier/c4008e70eea93e548533fd9f93a5a330

Example for running from terminal:
    python wass_launch.py -dr /home/mikapm/wass/ekok/exp/exp_20191209/

Reference:
    [1] Bergamasco et al. (2017): DOI: 10.1016/j.cageo.2017.07.001
"""

import os, sys
import glob
import shutil
import time
from multiprocessing import Pool
import subprocess
import numpy as np


def open_prog(command):
    """
    Function to run a command string with subprocess.
    Used in multiprocessing.Pool call for running in parallel.
    """
    return subprocess.run(command, shell=True, check=True)


class WASS_launcher():
    """
    Main WASS pipeline launcher class.

    See http://www.dais.unive.it/wass/documentation/getting_started.html
    for instructions on how to set up a new WASS experiment.
    """
    def __init__(self,
            process_cap = 4, # Max. number of parallel processes to use
            max_frames_to_match = 75,
            wass_dir = '/modules/bionic/user-apps/wass/8.10.2021/wass',
            exe_dir = None, # executables dir
            data_root = '/home/mikapm/wass/test/WASS_TEST/W07', # experiment dir
            overwrite_outdir = False, # overwrite existing output dir?
            img_type = 'tif', # Input image file extension: tif, png or jpg
            # Remove large files from output folders after;
            # -1 -> keep all files.
            delete_excessive_from = -1,
            ):

        self.process_cap = process_cap
        self.max_frames_to_match = max_frames_to_match
        self.wass_dir = wass_dir

        if not os.path.isdir(self.wass_dir):
            print('WASS root directory {} not found, \n'
                    'trying with $WASS_ROOT environment '
                    'variable. \n'.format(self.wass_dir))
            self.wass_dir = os.environ['WASS_ROOT']

        if exe_dir is None:
            self.exe_dir = os.path.join(self.wass_dir, 'dist', 'bin')
        else:
            self.exe_dir = exe_dir

        # Executables
        if os.name == 'nt':
            # Windows OS
            self.PREPARE_EXE = os.path.join(self.exe_dir, 'wass_prepare.exe')
            self.MATCH_EXE = os.path.join(self.exe_dir, 'wass_match.exe')
            self.AUTOCAL_EXE = os.path.join(self.exe_dir, 'wass_autocalibrate.exe')
            self.STEREO_EXE = os.path.join(self.exe_dir, 'wass_stereo.exe')
            self.ENV_SET = ''
        else:
            self.PREPARE_EXE = os.path.join(self.exe_dir, 'wass_prepare')
            self.MATCH_EXE = os.path.join(self.exe_dir, 'wass_match')
            self.AUTOCAL_EXE = os.path.join(self.exe_dir, 'wass_autocalibrate')
            self.STEREO_EXE = os.path.join(self.exe_dir, 'wass_stereo')
            self.ENV_SET = 'LD_LIBRARY_PATH="" && '

        # Define directories and files
        self.data_root = data_root # Work directory
        self.config_dir = os.path.join(self.data_root, 'config')
        self.matcher_config = os.path.join(self.config_dir, 'matcher_config.txt')
        self.stereo_config = os.path.join(self.config_dir, 'stereo_config.txt')
        self.input_c0_dir =  os.path.join(self.data_root, 'input/cam0')
        self.input_c1_dir =  os.path.join(self.data_root, 'input/cam1')
        self.out_dir =  os.path.join(self.data_root, 'out')

        self.overwrite_outdir = overwrite_outdir
        # Input image file extension
        self.img_type = img_type
        # Delete excessive output files starting at output workingdir:
        self.delete_excessive_from = delete_excessive_from

        # Sanity checks
        assert os.path.isdir(self.exe_dir), ('%s does not exist.'%self.exe_dir)
        assert os.path.isfile(self.PREPARE_EXE), ('%s does not exist.'%self.PREPARE_EXE)
        assert os.path.isfile(self.MATCH_EXE), ('%s does not exist.'%self.MATCH_EXE)
        assert os.path.isfile(self.AUTOCAL_EXE), ('%s does not exist.'%self.AUTOCAL_EXE)
        assert os.path.isfile(self.STEREO_EXE), ('%s does not exist.'%self.STEREO_EXE)
        assert os.path.isdir(self.data_root), ('%s does not exist.'%self.data_root)
        assert os.path.isdir(self.config_dir), ('%s does not exist.'%self.config_dir)
        assert os.path.isdir(self.input_c0_dir), ('%s does not exist.'%self.input_c0_dir)
        assert os.path.isdir(self.input_c1_dir), ('%s does not exist.'%self.input_c1_dir)

        # List cam0 frames
        cam0_frames = sorted(glob.glob(
            os.path.join(self.input_c0_dir, ('*{}.{}'.format('01', self.img_type)))))
        # List cam1 frames
        cam1_frames = sorted(glob.glob(
            os.path.join(self.input_c1_dir, ('*{}.{}'.format('02', self.img_type)))))
        # Save input frame names (and working directories wd) to dictionary
        if len(cam0_frames) == len(cam1_frames):
            self.input_frames = dict(Cam0=cam0_frames, Cam1=cam1_frames, wd=[])
            self.n_frames = len(self.input_frames['Cam0'])
            print('{} stereo frames found. \n'.format(self.n_frames))
            # Define working directories
            for i in range(self.n_frames):
                self.input_frames['wd'].append(os.path.join(
                    self.out_dir, '%06d_wd'%i))
        else:
            raise ValueError('Input directories contain an unequal number '
                    'of images.')


    def run_wass_prepare(self):
        """
        - Make output directory (overwrite/not?)
        - Run wass_prepare executable
        """

        # Make output directory if requested
        if os.path.exists(self.out_dir) and self.overwrite_outdir is True:
            print('{} already exists, removing it\n'.format(self.out_dir))
            shutil.rmtree(self.out_dir)
        elif os.path.exists(self.out_dir) and self.overwrite_outdir is False:
            print('{} exists, set overwrite_outdir to True to continue \n'.format(
                self.out_dir)
                )
            sys.exit()
        print('Making directory {} \n'.format(self.out_dir))
        os.mkdir(self.out_dir)

        # Run WASS prepare
        print('**************************************')
        print('** Running wass_prepare          *****')
        print('**************************************\n')

        # Initialise Pool with number of processes
        p = Pool(self.process_cap)

        #processes = []
        commands = []
        for i in range(self.n_frames):
            cmd = " ".join([
                    self.ENV_SET, 
                    self.PREPARE_EXE,
                    '--workdir', self.input_frames['wd'][i],
                    '--calibdir', self.config_dir,
                    '--c0', self.input_frames['Cam0'][i],
                    '--c1', self.input_frames['Cam1'][i],
                    ])
            commands.append(cmd)
        # Process in parallel
        print('Processing in parallel.. \n')
        p.map(open_prog, commands)
        p.close()
        p.terminate()
        p.join()
        

    def run_wass_match(self):
        """
        WASS match is used to match features in a subset of the input
        images. This forms the 1st part of the automatic extrinsic 
        stereo calibration.
        """

        # If more input frames than self.max_frames_to_match, pick random
        # frames to use in matching
        if self.n_frames > self.max_frames_to_match:
            frames_to_match = np.random.permutation(self.n_frames)
            frames_to_match = frames_to_match[:self.max_frames_to_match]
            frames_to_match.sort()
        else:
            frames_to_match = np.arange(self.n_frames)

        # Run WASS match (extrinsic calibration pt. 1)
        print('**************************************')
        print('** Running wass_match            *****')
        print('**************************************\n')

        # Initialise Pool with number of processes
        p = Pool(self.process_cap)

        #processes = []
        commands = []
        for i in range(len(frames_to_match)):
            ii = frames_to_match[i]
            cmd = " ".join([
                    self.ENV_SET, 
                    self.MATCH_EXE,
                    self.matcher_config,
                    self.input_frames['wd'][ii],
                    ])
            commands.append(cmd)
        print('Processing in parallel ... \n')
        p.map(open_prog, commands)
        p.close()
        p.terminate()
        p.join()


    def run_wass_autocalibrate(self):
        """
        WASS autocalibrate uses the matches found by wass_match
        to finish the automatic extrinsic calibration
        """

        # Create workspaces file
        workspaces = os.path.join(self.out_dir, 'workspaces.txt')
        with open(workspaces, 'w') as fid:
            for l in range(self.n_frames):
                fid.write(self.input_frames['wd'][l] + '\n')

        # Run WASS autocalibrate (extrinsic calibration pt. 2)
        print('**************************************')
        print('** Running wass_autocalibrate    *****')
        print('**************************************\n')

        # Path to workspaces.txt file
        workspaces = os.path.join(self.out_dir, 'workspaces.txt')
        command_str = " ".join([
            self.ENV_SET, 
            self.AUTOCAL_EXE,
            workspaces, 
            ])
        assert subprocess.run(
                command_str, shell=True).returncode == 0, ('component '
                        'exited with non-zero return code')


    def run_wass_stereo(self):
        """
        Runs the WASS dense stereo reconstruction.
        """

        # Run WASS stereo (dense stereo reconstruction)
        print('**************************************')
        print('** Running wass_stereo           *****')
        print('**************************************\n')

        # Initialise Pool with number of processes
        p = Pool(self.process_cap)

        #processes = []
        commands = [] # List for command strings
        for i in range(self.n_frames):
            cmd = " ".join([
                self.ENV_SET, self.STEREO_EXE,
                self.stereo_config, 
                self.input_frames['wd'][i],
                ])
            commands.append(cmd)
        print('Processing in parallel.. \n')
        p.map(open_prog, commands)
        p.close()
        p.terminate()
        p.join()


    def delete_excessive_output(self):
        """
        By default the WASS pipeline produces a large amount 
        of images and other large files. Run this function
        to delete these files from all but the N first
        output working directories, where N is defined in
        self.delete_excessive_from.
        """

        N = self.delete_excessive_from

        if self.n_frames <= N or N==-1:
            print('Nothing to remove ...')
            return 1

        # List filenames to be removed
        files_to_remove = [
                #'mesh.ply',
                # The following three are only generated by wass_match,
                # so they don't exist in all output directories
                #'plane_refinement_inliers.xyz',
                #'00000000_features.png',
                #'00000001_features.png',
                'undistorted/00000000.png',
                'undistorted/00000001.png',
                'undistorted/00000000_P0.png',
                'undistorted/00000001_P1.png',
                'undistorted/R0.png',
                'undistorted/R1.png',
                ]

        # Loop over all output working directories after N and remove files
        for outdir in self.input_frames['wd'][N:]:
            print('Removing excessive files from {}'.format(outdir))
            [os.remove(os.path.join(outdir,f)) for f in files_to_remove]


    def average_planes(self):
        """
        Reads plane.txt files in each work output folder 
        defined in self.input_frames['wd'] and averages 
        them into an average plane file plane_avg.txt in 
        the self.data_root directory
        """
        # Allocate space for all planes
        planes = np.zeros((self.n_frames, 4))
        # Loop through working direcories and read planes
        print('Averaging planes .......')
        for i,wd in enumerate(self.input_frames['wd']):
            plane_file = os.path.join(wd, 'plane.txt')
            # Read plane into array
            with open(plane_file, 'r') as pf:
                planes[i,:] = np.loadtxt(pf, dtype=float)
        # Average all planes
        plane_avg = np.mean(planes, axis=0)
        # Save txt
        avg_file = os.path.join(self.data_root, 'plane_avg.txt')
        np.savetxt(avg_file, plane_avg)
            

if __name__ == '__main__':
    from argparse import ArgumentParser
    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-pc", 
                help=("Number of processors to use in parallelisation"),
                type=int,
                default=4,
                )
        parser.add_argument("-mftm", 
                help=(" Max. number of frames to match with wass_match"),
                type=int,
                default=75,
                )
        parser.add_argument("-dr", 
                help=("Path to data_root (work) directory"),
                type=str,
                default='/home/mikapm/wass/test/WASS_TEST/W07',
                )
        parser.add_argument("-overwrite", 
                help=("Overwrite existing output dir?"),
                type=int,
                choices=[0,1],
                default=0,
                )
        parser.add_argument("-type", 
                help=("Image format (file extension)"),
                type=str,
                choices=['tif', 'png', 'jpg'],
                default='tif',
                )
        parser.add_argument("-defn", 
                help=("Delete excessive files after Nth output "
                    "working directory. Set to -1 to keep all "
                    "files"),
                type=int,
                default=-1,
                )
        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Create WASS launcer object using default or input arguments
    WL = WASS_launcher(
            process_cap = args.pc,
            max_frames_to_match = args.mftm,
            data_root = args.dr,
            overwrite_outdir = bool(args.overwrite),
            img_type = args.type,
            delete_excessive_from = args.defn,
            )
    
    # Time the entire processing
    start_process = time.time()

    # Run WASS prepare
    start = time.time() # measure time 
    WL.run_wass_prepare()
    end = time.time()
    print('Done in %.2f seconds. \n' % (end-start))

    # Run WASS match
    start = time.time() 
    WL.run_wass_match()
    end = time.time()
    print('Done in %.2f seconds. \n' % (end-start))

    # Run WASS autocalibrate
    start = time.time() 
    WL.run_wass_autocalibrate()
    end = time.time()
    print('Done in %.2f seconds. \n' % (end-start))

    # Run WASS stereo
    start = time.time() 
    WL.run_wass_stereo()
    end = time.time()
    print('Done in %.2f seconds. \n' % (end-start))

    # Remove excessive output files
    WL.delete_excessive_output()

    # Average planes
    WL.average_planes()

    end_process = time.time()
    print('Full processing done in %.2f seconds. \n' % 
            (end_process-start_process))


