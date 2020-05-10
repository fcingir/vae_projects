#!/usr/bin/env python

#import scipy.misc
import imageio
import numpy as np
import random
import tensorflow as tf
import _pickle as cPickle
import matplotlib


def get_image(image_path, image_size, is_crop=False):
    return transform(imread(image_path), image_size, is_crop)

def transform(image, npx=32, is_crop=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w=None, resize_w=32):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    #return imageio.imresize(x[j:j+crop_h, i:i+crop_w],
    #                         [resize_w, resize_w])
    return x
    
def imread(path):
    readimage = imageio.imread(path).astype(np.float)
    return readimage

def merge_color(images, size):
    print(images.shape, images.shape)
    #h, w = images.shape[1], images.shape[2]
    h, w = 32, 32
    img = np.zeros((h * np.array(size)[0], w * np.array(size)[1], 3))

    for idx, image in enumerate(images):
        #print('image shape=', image.shape, 'image', image)
        i = idx % size[1]
        j = int(idx / size[1])
        #print('i=', str(i), ' j= ', str(j), ' w =', str(w), ' h =', str(h), str(np.array(size)))  ## function threw an error, this to check
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img #transpose because the tensor placeholder requires this

def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def ims(name, img):
    # print img[:10][:10]
    #imageio.imwrite(img, cmin=0, cmax=1).save(name)   # original
    print('imgshape',img.shape)
    #imageio.imwrite(name, (img*255).astype(np.uint8))
    imageio.imwrite(name, (img*255).astype(np.uint8))
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
