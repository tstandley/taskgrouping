import torch.utils.data as data

from PIL import Image, ImageOps
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
import warnings
import torchvision.transforms as transforms

from multiprocessing import Manager


class TaskonomyLoader(data.Dataset):
    def __init__(self,
                 root,
                 label_set=['depth_zbuffer','normal','segment_semantic','edge_occlusion','reshading','keypoints2d','edge_texture'],
                 model_whitelist=None,
                 model_limit=None,
                 output_size=None,
                 convert_to_tensor=True,
                 return_filename=False,
                 half_sized_output=False,
                 augment=False):
        manager=Manager()
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
        
        #self.records = manager.list(self.records)
        self.label_set = label_set
        self.output_size = output_size
        self.half_sized_output=half_sized_output
        self.convert_to_tensor = convert_to_tensor
        self.return_filename=return_filename
        self.to_tensor = transforms.ToTensor()
        self.augment = augment
        
        if augment == "aggressive":
            print('Data augmentation is on (aggressive).')
        elif augment:
            print('Data augmentation is on (flip).')
        else:
            print('no data augmentation')
        self.last = {}
    
    def process_image(self,im,input=False):
        output_size=self.output_size
        if self.half_sized_output and not input:
            if output_size is None:
                output_size=(128,128)
            else:
                output_size=output_size[0]//2,output_size[1]//2
        if output_size is not None and output_size!=im.size:
            im = im.resize(output_size,Image.BILINEAR)
        
        bands = im.getbands()
        if self.convert_to_tensor:
            if bands[0]=='L':
                im = np.array(im)
                im.setflags(write=1)
                im = torch.from_numpy(im).unsqueeze(0)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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
        with torch.no_grad():
            file_name=self.records[index]
            save_filename = file_name
            
            flip_lr = (random.randint(0,1) > .5 and self.augment)

            flip_ud = (random.randint(0,1) > .5 and (self.augment=="aggressive"))

            
            
            
            pil_im = Image.open(file_name)
            
            if flip_lr:
                pil_im = ImageOps.mirror(pil_im)
            if flip_ud:
                pil_im = ImageOps.flip(pil_im)
            
            im = self.process_image(pil_im,input=True)
            
            error=False
    
            ys = {}
            mask = None
            to_load = self.label_set
            if len(set(['edge_occlusion','normal','reshading','principal_curvature']).intersection(self.label_set))!=0:
                if os.path.isfile(file_name.replace('rgb','mask')):
                    to_load.append('mask')
                elif 'depth_zbuffer' not in to_load:
                    to_load.append('depth_zbuffer')

            for i in to_load:
                if i=='mask' and mask is not None:
                    continue
                
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
                if flip_lr:
                    try:
                        yim = ImageOps.mirror(yim)
                    except:
                        pass
                if flip_ud:
                    try:
                        yim = ImageOps.flip(yim)
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
                    yim-=56.0248
                    yim/=239.1265
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
                    if flip_lr:
                        yim[0]*=-1.0
                    if flip_ud:
                        yim[1]*=-1.0
                elif i == 'reshading':
                    yim=yim.mean(dim=0,keepdim=True)
                    yim-=.4962
                    yim/=0.2846
                    #print('reshading',yim.shape,yim.max(),yim.min())
                elif i == 'principal_curvature':
                    yim=yim[:2]
                    yim-=torch.tensor([0.5175, 0.4987]).view(2,1,1)
                    yim/=torch.tensor([0.1373, 0.0359]).view(2,1,1)
                    #print('principal_curvature',yim.shape,yim.max(),yim.min())
                elif i == 'mask':
                    mask=yim.bool()
                    yim=mask
                
                ys[i] = yim



            if mask is not None:
                ys['mask']=mask
            
            # print(self.label_set)
            # print('rgb' in self.label_set)
            if not 'rgb' in self.label_set:
                ys['rgb']=im
            
            if self.return_filename:
                return im, ys, file_name
            else:
                return im, ys


    def __len__(self):
        return (len(self.records))

def show(im, ys):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30,30))
    plt.subplot(4,3,1).set_title('RGB')
    im = im.permute([1,2,0])
    plt.imshow(im)
    #print(im)
    #print(ys)
    for i, y in enumerate(ys):
        yim=ys[y]
        plt.subplot(4,3,2+i).set_title(y)
        if y=='normal':
            yim+=1
            yim/=2
        if yim.shape[0]==2:
            yim = torch.cat([yim,torch.zeros((1,yim.shape[1],yim.shape[2]))],dim=0)
        yim = yim.permute([1,2,0])
        yim = yim.squeeze()
        plt.imshow(np.array(yim))
        
    
    plt.show()

def test():
    loader = TaskonomyLoader(
        '/home/tstand/Desktop/lite_taskonomy/',
        label_set=['normal','reshading','principal_curvature','edge_occlusion','depth_zbuffer'],
        augment='aggressive')

    totals= {}
    totals2 = {}
    count = {}
    indices= list(range(len(loader)))
    random.shuffle(indices)
    for data_count, index in enumerate(indices):
        im, ys=loader[index]
        show(im,ys)
        mask = ys['mask']
        #mask = ~mask
        print(index)
        for i, y in enumerate(ys):
            yim=ys[y]
            yim = yim.float()
            if y not in totals:
                totals[y]=0
                totals2[y]=0
                count[y]=0
        
            totals[y]+=(yim*mask).sum(dim=[1,2])
            totals2[y]+=((yim**2)*mask).sum(dim=[1,2])
            count[y]+=(torch.ones_like(yim)*mask).sum(dim=[1,2])
            
            #print(y,yim.shape)
            std = torch.sqrt((totals2[y]-(totals[y]**2)/count[y])/count[y])
            print(data_count,'/',len(loader),y,'mean:',totals[y]/count[y],'std:',std)

def output_mask(index,loader):
    from matplotlib import pyplot as plt
    filename=loader.records[index]
    filename=filename.replace('rgb','mask')
    filename=filename.replace('/intel_nvme/taskonomy_data/','/run/shm/')
    if os.path.isfile(filename):
        return

    
    print(filename)
    

    x,ys = loader[index]
    
    mask =ys['mask']
    mask=mask.squeeze()
    mask_im=Image.fromarray(mask.numpy())
    mask_im = mask_im.convert(mode='1')
    # plt.subplot(2,1,1)
    # plt.imshow(mask)
    # plt.subplot(2,1,2)
    # plt.imshow(mask_im)
    # plt.show()
    path, _ = os.path.split(filename)
    os.makedirs(path,exist_ok=True)
    mask_im.save(filename,bits=1,optimize=True)
    
    




def get_masks():
    import multiprocessing


    loader = TaskonomyLoader(
        '/intel_nvme/taskonomy_data/',
        label_set=['depth_zbuffer'],
        augment=False)

    indices= list(range(len(loader)))
    
    random.shuffle(indices)


    for count,index in enumerate(indices):
        print(count,len(indices))
        output_mask(index,loader)

        

        

if __name__ == "__main__":
    test()
    #get_masks()
