#!/usr/bin/env python
#******************************************************************************
#  Name:     cnnS2class.py (colab version)
#  Purpose:  Use CNN to find target patches in an S2 image
#  Usage (from command line):
#    python cnnS2class.py  [options] filename
#
#  Copyright (c) 2021 Mort Canty

import sys, os, time, getopt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

def main():
    global model

    usage = '''
Usage:
--------------------------------------

Deep learning classification of S2 images

python %s [OPTIONS] filename

Options:
  -h            this help
  -m  <string>  path to stored model
  -d  <list>    spatial subset [x,y,width,height]
  -t  <float>   class probability threshold (default 0.7)
  -s  <int>     stride (default 64)

Classes:
                    0 'AnnualCrop',
                    1 'Forest',
                    2 'HerbaceousVegetation',
                    3 'Highway',
                    4 'Industrial',
                    5 'Pasture',
                    6 'PermanentCrop',
                    7 'Residential',
                    8 'River',
                    9 'SeaLake'

Assumes S2 image bands are B2 B3 B4 only

  -------------------------------------'''%sys.argv[0]

    options,args = getopt.getopt(sys.argv[1:],'hnm:t:d:s:v:')
    model_path = 'data/eurosat_model.h5'
    dims = None
    stride = 64
    thresh = 0.7

    for option, value in options:
        if option == '-h':
            print(usage)
            return
        elif option == '-m':
            model_path = value
        elif option == '-d':
            dims = eval(value)
        elif option == '-t':
            thresh = eval(value)
        elif option == '-s':
            stride = eval(value)
    if len(args) != 1:
        print( 'Incorrect number of arguments' )
        print(usage)
        sys.exit(1)

    gdal.AllRegister()

#  load the trained CNN model
    model = keras.models.load_model(model_path)

    infile = args[0]
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_cnn'+ext

    start = time.time()

    inDataset = gdal.Open(infile, GA_ReadOnly)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize

    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0

#  read in as RGB
    G = np.zeros((rows,cols,3))
    for b in range(3):
        band = inDataset.GetRasterBand(3-b)
        tmp = (band.ReadAsArray(x0,y0,cols,rows)/3000)*255.
        G[:,:,b] = np.where(tmp>255,255,tmp)

#   Classify numpy image array in 64x64 patches '''
    h, w, _ = G.shape
    hk = h // stride
    wk = w // stride
    cmap = np.zeros((hk,wk,2))
    for i in tqdm(range(hk), desc="Classifying..."):
        for j in range(wk):
            Gs = G[max(i*stride-32,0):i*stride+32,max(j*stride-32,0):j*stride+32,:]
            patch = tf.image.resize(Gs,[224,224])
            patch =  keras.applications.xception.preprocess_input(patch)
            res = model.predict(np.array([patch]), verbose = 0)
            cmap[i,j,:] = [np.argmax(res), np.max(res)] #pick category and save score

#  mark the target patches
    Gmap = np.zeros((rows,cols,4))
    Gmap[:,:,0:3] = G.copy()
    for i in range(hk):
        for j in range(wk):
            if cmap[i,j,1] > thresh:
                Gmap[i*stride:(i+1)*stride,j*stride:(j+1)*stride,3] = cmap[i,j,0]

#  write to disk
    driver =  gdal.GetDriverByName('GTiff')
    outDataset = driver.Create(outfile,cols,rows,4,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)
    for b in range(4):
        outBand = outDataset.GetRasterBand(b+1)
        outBand.WriteArray(Gmap[:,:,b],0,0)
        outBand.FlushCache()
    inDataset = None
    outDataset = None
    print('\nMap written to: %s'%outfile)
    print('Elapsed time: %s'%str(time.time()-start))

if __name__ == '__main__':
    main()
