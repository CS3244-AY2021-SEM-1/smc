'''
Contains the ImageDataLoader class
- ImageDataLoader returns an iterable object
- Each element in the iterable object is a dictionary containing:
    - image: the image
    - gt_density
- Instantiate a new ImageDataLoader object using ImageDataLoader(data_path, shuffle=False, pre_load=False)
    - data_path: the path to the consolidated .h5 files
    - shuffle (default=False): shuffles the .h5 files
    - pre-load (default=False): if true, all training and validation images are loaded into CPU RAM for faster processing. 
                                This avoids frequent file reads. Use this only for small datasets.
'''

import numpy as np
import cv2
import os
import random
import pandas as pd
import h5py
import ast


class ImageDataLoader():
    def __init__(self, data_path, shuffle=False, pre_load=False, num_pool=None, size=300):

        self.data_path = data_path
        self.pre_load = pre_load
        self.shuffle = shuffle
        self.num_pool = num_pool
        if shuffle: random.seed(2468)

        self.data_files = [os.path.join(data_path, filename) for filename in os.listdir(data_path)
                           if os.path.isfile(os.path.join(data_path, filename))][0:size]

        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = list(range(0, self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                blob = {}
                f = h5py.File(fname, "r")
                
                # target shape
                target_shape = (720, 1280)
                divide = 2**num_pool
                gt_target_shape = (720//divide, 1280//divide)

                img = f['image'][()]
                den = f['density'][()]
                metadata = f['metadata'][()]
                
                # resizing with cv2
                img_resized = cv2.resize(img, target_shape, interpolation = cv2.INTER_CUBIC)
                gt_resized = cv2.resize(den, gt_target_shape, interpolation = cv2.INTER_CUBIC)

                # if BW, skip
                if img_resized.shape == (target_shape[1], target_shape[0]): continue
                if len(img_resized.shape) == 2: continue

                blob['data'] = img.reshape((1, 3, img.shape[0], img.shape[1]))
                blob['gt_density'] = den.reshape((1, 1, den.shape[0], den.shape[1]))
                blob['metadata'] = ast.literal_eval(metadata)
                
                self.blob_list[idx] = blob

                idx += 1
                if idx % 100 == 0:
                    print(f'Loaded {idx} / {self.num_samples} files')

            print(f'Completed Loading {idx} files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list
        num_pool = self.num_pool

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fname = files[idx]
                blob = {}
                f = h5py.File(fname, "r")

                img = f['image'][()]
                den = f['density'][()]
                metadata = f['metadata'][()]

                # target shape
                target_shape = (720, 1280)
                divide = 2**num_pool
                gt_target_shape = (720//divide, 1280//divide)

                # resizing with cv2
                img_resized = cv2.resize(img, target_shape, interpolation = cv2.INTER_LINEAR)
                gt_resized = cv2.resize(den, gt_target_shape, interpolation = cv2.INTER_LINEAR)

                # if BW, skip
                if img_resized.shape == (target_shape[1], target_shape[0]): continue
                if len(img_resized.shape) == 2: continue
                    
                    
                blob['data'] = img_resized.reshape(1, 3, target_shape[0], target_shape[1])
                blob['gt_density'] = gt_resized.reshape(1, 1, gt_target_shape[0], gt_target_shape[1]) * (4**num_pool)
                blob['metadata'] = ast.literal_eval(metadata)
                
                self.blob_list[idx] = blob

            yield blob

    def get_num_samples(self):
        return self.num_samples
    
    def get_test_input(self, num_pool=2, index=0):
        fname = self.data_files[index]
        blob = {}
        f = h5py.File(fname, "r")

        img = f['image'][()]
        den = f['density'][()]
        metadata = f['metadata'][()]

        # target shape
        target_shape = (720, 1280)
        divide = 2**num_pool
        gt_target_shape = (720//divide, 1280//divide)

        # resizing with cv2
        img_resized = cv2.resize(img, target_shape, interpolation = cv2.INTER_CUBIC)
        gt_resized = cv2.resize(den, gt_target_shape, interpolation = cv2.INTER_CUBIC)

        blob['data'] = img_resized.reshape(1, 3, target_shape[0], target_shape[1])
        blob['gt_density'] = gt_resized.reshape(1, 1, gt_target_shape[0], gt_target_shape[1]) * (4**num_pool)
        blob['metadata'] = ast.literal_eval(metadata)
        
        return blob


