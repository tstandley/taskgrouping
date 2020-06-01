import torch.utils.data as data

from PIL import Image
import os
import os.path
import zipfile as zf
import io
import logging
import random
import copy
import numpy as np
import time
import torch

import multiprocessing

import torchvision.transforms as transforms



class TaskonomyLoader(data.Dataset):
    def __init__(self,
                 root,
                 label_set=['segment_semantic','depth_zbuffer','normal','edge_occlusion','reshading','keypoints2d','edge_texture'],
                 model_whitelist=None,
                 model_limit=None,
                 output_size=None,
                 convert_to_tensor=True,
                 return_filename=False, 
                 augment=False,
                 partition=False
                 ):
        self.partition = partition
        self.label_set = label_set
        self.root = root
        self.model_limit=model_limit
        self.records=[]
        if model_whitelist is None:
            self.model_whitelist=None
        else:
            self.model_whitelist = set()
            with open(model_whitelist) as f:
                for line in f:
                    self.model_whitelist.add(line.strip())
        
        for i,(where, subdirs, files) in enumerate(os.walk(os.path.join(root,'rgb'))):
            if subdirs!=[]: continue
            model = where.split('/')[-1]
            if self.model_whitelist is None or model in self.model_whitelist:
                full_paths = [os.path.join(where,f) for f in files]
                if isinstance(model_limit, tuple):
                    full_paths.sort()
                    full_paths = full_paths[model_limit[0]:model_limit[1]]
                elif model_limit is not None:
                    full_paths.sort()
                    full_paths = full_paths[:model_limit]
                self.records+=full_paths

        


        if not self.partition:
            self.records = list( map( lambda record: {task: record for task in self.label_set}, self.records) )
        else:
            def chunks(lst):
                n = len(self.label_set)
                """Yield successive n-sized chunks from lst. Source: https://stackoverflow.com/a/312464"""
                for i in range(n, len(lst), n):
                    yield lst[i-n : i]
            self.records = list( chunks( self.records ) )
            
            def convert_list_to_dict(record):
                result = {}
                for i in range( len( self.label_set ) ):
                    task = self.label_set[i]
                    result[task] = record[i]
                return result
            self.records = list( map( convert_list_to_dict, self.records ) )
            

        self.output_size = output_size
        self.convert_to_tensor = convert_to_tensor
        self.return_filename=return_filename
        self.to_tensor = transforms.ToTensor()
        self.augment = augment
        if augment:
            print('Data augmentation is on (flip).')
        self.last = {}
    
    def process_image(self,im):
        if self.output_size is not None and self.output_size!=im.size:
            im = im.resize(self.output_size,Image.BILINEAR)
        
        bands = im.getbands()
        if self.convert_to_tensor:
            if bands[0]=='L':
                im = np.array(im)
                im = torch.from_numpy(im).unsqueeze(0)
            else:
                im = self.to_tensor(im)

        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is an uint8 matrix of integers with the same width and height.
        If there is an error loading an image or its labels, simply return the previous example.
        """
        
        ims = {}
        ys = {}
        mask = None

        for i in self.label_set:

            file_name=self.records[index][i]
            save_filename = file_name
        
            flip = (random.randint(0,1) > .5 and self.augment)
        
            pil_im = Image.open(file_name)
        
            if flip:
                pil_im = pil_im.transpose(Image.FLIP_LEFT_RIGHT)
        
            im = self.process_image(pil_im)
        
            error=False

            yfilename = file_name.replace('rgb',i)
            try:
                yim = Image.open(yfilename)
            except:
                yim = self.last[i].copy()
                error = True
            if (i in self.last and yim.getbands() != self.last[i].getbands()) or error:
                yim = self.last[i].copy()
            try:
                self.last[i]=yim.copy()
            except:
                pass
            if flip:
                try:
                    yim = yim.transpose(Image.FLIP_LEFT_RIGHT)
                except:
                    pass
            try:
                yim = self.process_image(yim)
            except:
                yim = self.last[i].copy()
                yim = self.process_image(yim)

            if i == 'depth_zbuffer':
                yim = yim.float()
                mask = yim < (2**13)
                yim-=1500.0
                yim/= 1000.0
            elif i == 'edge_occlusion':
                yim = yim.float()
                yim-=50.0
                yim/=100.0
            elif i == 'keypoints2d':
                yim = yim.float()
                yim-=50.0
                yim/=100.0
            elif i == 'edge_texture':
                yim = yim.float()
                yim-=718.0
                yim/=1070.0
            elif i == 'normal':
                yim = yim.float()
                yim -=.5
                yim *=2.0
                if flip:
                    yim[0]*=-1.0
                else:
                    pass
                
            ys[i] = yim
            ims[i] = im
            
        if mask is not None:
            ys['mask']=mask
        
        return ims, ys


    def __len__(self):
        return (len(self.records))
