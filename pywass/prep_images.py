#!/usr/bin/env python3
"""
Script for taking a set of .raw images
from EKOK and converting + renaming them
to be compatible with the WASS stereo
wave software.

Takes as first input the path to a directory
containing the images. All files with .raw
extension will be converted (not replaced).

If input photos not in true .raw format,
but instead jpg, png, etc., give second input 
arg "notraw".
"""

import sys
import glob
import os
import numpy as np
import cv2
from argparse import ArgumentParser

def parse_args(**kwargs):
    parser = ArgumentParser()
    parser.add_argument("datadir", 
            help=("Path to data directory"),
            type=str,
            )
    parser.add_argument("-outdir", 
            help=("Path to output directory. If not "
                "specified, will use datadir."),
            type=str,
            )
    parser.add_argument("-file_ext", 
            help=("File extension"
                ),
            type=str,
            choices=['raw', 'png', 'tif', 'tiff', 'jpg'],
            default='raw',
            )
    return parser.parse_args(**kwargs)

# Call args parser to create variables out of input arguments
args = parse_args(args=sys.argv[1:])

# Directory of images from input
imDir = args.datadir
imNamesLeft = glob.glob(os.path.join(imDir, ('*358.' + args.file_ext)))
imNamesRight = glob.glob(os.path.join(imDir, ('*400.' + args.file_ext)))
# Sort files
imNamesLeft.sort()
imNamesRight.sort()
print(imNamesLeft[0])

# Determine image size by opening one image in left image
# directory. Assuming all images in directory have the same
# resolution.
fN_0 = imNamesLeft[0]
with open(fN_0, 'rb') as f:
    arr_0 = np.fromfile(f, dtype=np.uint8)
    if arr_0.size == (2048*2448):
        imRow = 2048
        imCol = 2448
    elif arr_0.size == (1024*1224):
        imRow = 1024
        imCol = 1224
    else:
        raise ValueError('Unknown image resolution.')

# Define output directories
if args.outdir is None:
    outPath = args.datadir
else:
    outPath = args.outdir
# Make output directories for cam0 and cam1
cam0Dir = os.path.join(outPath, 'cam0')
cam1Dir = os.path.join(outPath, 'cam1')
if not os.path.isdir(cam0Dir):
    os.mkdir(cam0Dir)
if not os.path.isdir(cam1Dir):
    os.mkdir(cam1Dir)

# Loop over files in imNamesLeft:
for i,fN in enumerate(imNamesLeft):
    # Read raw data into array
    print(fN)

    # Set output filename to comply with WASS requirements
    basePath, rawFile = os.path.split(fN)
    rawFileSplit = rawFile.split('-')
    date = rawFileSplit[0]
    timeStampInt = int(rawFileSplit[0][-6:] + rawFileSplit[2])
    camSerNo = rawFileSplit[3].split('.')[0]
    # WASS filename format:
    # <sequence_number (6 digits)>_<timestamp (13 digits)>_<camera number (2 digits)>.tif
    seqNo = ('%06d' % i)
    timeStamp = ('%013d' % timeStampInt)
    if camSerNo == '16048358':
        camNo = '01'
    elif camSerNo == '16048400':
        camNo = '02'
    else:
        raise ValueError('Unknown camera serial number %s!' % camSerNo)
    wassFileName = (seqNo + '_' + timeStamp + '_' + camNo + '.tif')
    outFile = os.path.join(cam0Dir, wassFileName)
    # Check if tif file already exists
    if os.path.isfile(outFile):
        print('File {} exists, skipping to next one ...\n'.format(
            fN))
        continue

    # Check if file size is 0
    if os.stat(fN).st_size == 0:
        print('Skipping file {}, no content ... \n'.format(fN))
        continue
    if args.file_ext != 'raw':
        im = cv2.imread(fN, 0)
    else:
        with open(fN, 'rb') as f:
            arr = np.fromfile(f, dtype=np.uint8,count=imRow*imCol)
            im = arr.reshape((imRow,imCol))
    if len(im.shape) > 2:
        # Throw away useless color channels
        im = im[:,:,0]

    # Save image as .tif
    cv2.imwrite(outFile, im)

# Loop over files in imNamesRight:
for i,fN in enumerate(imNamesRight):
    # Read raw data into array
    print(fN)

    # Set output filename to comply with WASS requirements
    basePath, rawFile = os.path.split(fN)
    rawFileSplit = rawFile.split('-')
    date = rawFileSplit[0]
    timeStampInt = int(rawFileSplit[0][-6:] + rawFileSplit[2])
    camSerNo = rawFileSplit[3].split('.')[0]
    # WASS filename format:
    # <sequence_number (6 digits)>_<timestamp (13 digits)>_<camera number (2 digits)>.tif
    seqNo = ('%06d' % i)
    timeStamp = ('%013d' % timeStampInt)
    if camSerNo == '16048358':
        camNo = '01'
    elif camSerNo == '16048400':
        camNo = '02'
    else:
        raise ValueError('Unknown camera serial number %s!' % camSerNo)
    wassFileName = (seqNo + '_' + timeStamp + '_' + camNo + '.tif')
    outFile = os.path.join(cam1Dir, wassFileName)
    if os.path.isfile(outFile):
        print('File {} exists, skipping to next one ...\n'.format(
            fN))
        continue

    # Check if file size is 0
    if os.stat(fN).st_size == 0:
        print('Skipping file {}, no content ... \n'.format(fN))
        continue
    if args.file_ext != 'raw':
        im = cv2.imread(fN, 0)
    else:
        with open(fN, 'rb') as f:
            arr = np.fromfile(f, dtype=np.uint8,count=imRow*imCol)
            im = arr.reshape((imRow,imCol))
    if len(im.shape) > 2:
        # Throw away useless color channels
        im = im[:,:,0]

    # Save image as .tif
    cv2.imwrite(outFile, im)


