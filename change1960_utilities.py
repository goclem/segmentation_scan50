#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
'''

#%% MODULES

import geopandas
import numpy as np
import rasterio
import os
import re
import shutil

from itertools import compress
from matplotlib import pyplot
from numpy import random
from rasterio import features

#%% PATHS UTILITIES

paths = dict(
    data='../data_1960',
    images='../data_1960/images',
    labels='../data_1960/labels',
    models='../data_1960/models',
    predictions='../data_1960/predictions',
    statistics='../data_1960/statistics',
    figures='../data_1960/figures'
)

#%% FILES UTILITIES

def search_data(directory:str='.', pattern:str='.*tif$') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    return files

def identifiers(files:list, regex:bool=False, extension:str='tif') -> list:
    '''Extracts file identifiers'''
    identifiers = [os.path.splitext(os.path.basename(file))[0] for file in files]
    identifiers = [identifier[identifier.find('_') + 1:] for identifier in identifiers]
    identifiers.sort()
    if regex:
        identifiers = '({identifiers})\\.{extension}$'.format(identifiers='|'.join(identifiers), extension=extension)
    return identifiers

def filter_identifiers(files:list, filter:list) -> list:
    '''Filters by identifiers'''
    subset = np.isin(identifiers(files), identifiers(filter), invert=True)
    subset = list(compress(files, subset))
    return subset

def initialise_directory(directory:str, remove:bool=False):
    '''Initialises a directory'''
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(directory) and remove is True:
        shutil.rmtree(directory)
        os.mkdir(directory)

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str=None) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def rasterise(source, profile, attribute:str=None, dtype:str='uint8') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

#%% NUMPY UTILITIES

def images_to_blocks(images:np.ndarray, blocksize:tuple=(256, 256), shift:bool=False, mode:str='constant', constant_values:int=None) -> np.ndarray:
    '''Converts images to blocks of a given size'''
    # Initialises quantities
    nimages, imagewidth, imageheight, nbands = images.shape
    blockwidth, blockheight = blocksize
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Reshape images into blocks
    images = np.pad(images, ((0, 0), (padwidth, padwidth), (padheight, padheight), (0, 0)), mode=mode, constant_values=constant_values)
    blocks = images.reshape(nimages, nblockswidth, blockwidth, nblocksheight, blockheight, nbands).swapaxes(2, 3)
    blocks = blocks.reshape(-1, blockwidth, blockheight, nbands)
    return blocks

def blocks_to_images(blocks:np.ndarray, imagesize:tuple, shift:bool=False) ->  np.ndarray:
    '''Converts blocks to images of a given size'''
    # Initialises quantities
    nimages, imagewidth, imageheight, nbands = imagesize
    blockwidth, blockheight = blocks.shape[1:3]
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Converts blocks into images
    images = blocks.reshape(-1, nblockswidth, nblocksheight, blockwidth, blockheight, nbands).swapaxes(2, 3)
    images = images.reshape(-1, (imagewidth + (2 * padwidth)), (imageheight + (2 * padheight)), nbands)
    images = images[:, padwidth:imagewidth + padwidth, padheight:imageheight + padheight, :]
    return images

def not_empty(image, type:str='all', value:int=255):
    '''Checks for empty images'''
    test = np.equal(image, np.full(image.shape, value))
    if type == 'all': test = np.all(test)
    if type == 'any': test = np.any(test)
    test = np.invert(test)
    return test

def sample_split(images:np.ndarray, sizes:dict, seed:int=1) -> list:
    '''Splits the data multiple samples'''
    random.seed(seed)
    samples = list(sizes.keys())
    indexes = random.choice(samples, images.shape[0], p=list(sizes.values()))
    samples = [images[indexes == sample, ...] for sample in samples]
    return samples

#%% DISPLAY UTILITIES
    
def display(image:np.ndarray, title:str='', cmap:str='gray', path:str=None) -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    if path is not None:
        pyplot.savefig(path, dpi=300)
    else:
        pyplot.show()

def compare(images:list, titles:list=['Image'], cmaps:list=['gray'], path:str=None) -> None:
    '''Displays multiple images'''
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    if len(cmaps) == 1:
        cmaps = cmaps * nimage
    fig, axs = pyplot.subplots(nrows=1, ncols=nimage, figsize=(10, 10 * nimage))
    for ax, image, title, cmap in zip(axs.ravel(), images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    pyplot.tight_layout()
    if path is not None:
        pyplot.savefig(path, dpi=300)
    else:
        pyplot.show()

def display_history(history:dict, stats:list=['accuracy', 'loss']) -> None:
    '''Displays training history'''
    fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax, stat in zip(axs.ravel(), stats):
        ax.plot(history[stat])
        ax.plot(history[f'val_{stat}'])
        ax.set_title(f'Training {stat}', fontsize=15)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training sample', 'Validation sample'], frameon=False)
    pyplot.tight_layout(pad=2.0)
    pyplot.show()