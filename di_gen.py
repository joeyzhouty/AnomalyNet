#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
approximate rank pooling

input: X(image sequence for dynamic image): w*h*N 
output: DI_gray (uint8):w*h
    
"""

import numpy as np
from PIL import Image

## test code
X=np.zeros(shape=(480,640,20))

for im_index in range(20):
    img=Image.open('./data/'+'depth_'+'{:0>4d}'.format(im_index+140)+'.jpg')
    img=img.convert('L')
    X[:,:,im_index]=np.array(img)


## approximate rank pooling  
# input: X w*h*N image sequence
im_shape=np.shape(X)    
N=im_shape[2]

fw=np.zeros(shape=(1,1,N))
for x_index in range(N):
    fw[0,0,x_index]=2*(x_index+1)-N-1

Y=np.multiply(X,fw)

DI=np.zeros(shape=(im_shape[0],im_shape[1]))  
for y_index in range(N):
    tmp=Y[:,:,y_index]
    DI=DI+tmp

pixel_max=np.max(DI)
pixel_min=np.min(DI)

DI_gray=(DI-pixel_min)/(pixel_max-pixel_min)*255.
DI_gray=Image.fromarray(np.uint8(DI_gray))
DI_gray.show()
