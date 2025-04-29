#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Preprocessing for the Arthisto1960 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
'''

#%% MODULES
import geopandas

from histo1960_utilities import *
from os import path

paths = dict(
    images_raw='/Users/clementgorin/Dropbox/data/ign_scan50',
    labels_raw='../shared_ras/training1960',
    images='../data_1960/images', 
    labels='../data_1960/labels'
)

#%% FORMATS IMAGES

srcfiles = search_data(paths['images_raw'], 'tif$')
for srcfile in srcfiles:
    print(path.basename(srcfile))
    outfile = path.basename(srcfile).replace('sc50', 'image')
    outfile = path.join(paths['images'], outfile)
    if not path.exists(outfile):
        os.system('gdal_translate -ot byte {srcfile} {outfile}'.format(srcfile=srcfile, outfile=outfile))

#%% FORMATS LABELS

# Builds file paths
srclabels = search_data(paths['labels_raw'], 'label_\\d{4}_\\d{4}\\.gpkg$')
srclabels = list(filter(re.compile('^(?!.*(incomplete|pending)).*').search, srclabels))
srcimages = search_data(paths['labels_raw'], identifiers(srclabels, regex=True))
outlabels = [path.join(paths['labels'], path.basename(file).replace('.gpkg', '.tif')) for file in srclabels]

# Rasterises label vectors
for srclabel, srcimage, outlabel in zip(srclabels, srcimages, outlabels):
    print(path.basename(outlabel))
    label = rasterise(srclabel, srcimage)
    write_raster(label, srcimage, outlabel)
del srclabels, srcimages, outlabels


