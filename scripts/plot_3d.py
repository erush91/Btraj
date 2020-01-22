#!/usr/bin/env python

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import cv2

from mayavi.mlab import *

filepath = '/home/erush91/.ros/test_fm3d.txt'

with open(filepath) as fp:
    lines = fp.readlines()

name    = lines[0].rstrip()
dim     = float(lines[1].rstrip())
res     = float(lines[2].rstrip())
x_size  = int(lines[3].rstrip())
y_size  = int(lines[4].rstrip())
z_size  = int(lines[5].rstrip())
data    = lines[6:]

length_data = len(data)

for i in range(length_data):
    data[i] = data[i].rstrip()

    if(data[i] == 'inf'):
        data[i] = 0
    
    data[i] = float(data[i])

data = np.asarray(data)

print 'name:', name
print 'res:', res
print 'x_size:', x_size
print 'y_size:', y_size
print 'z_size:', z_size
print 'length_data:', length_data

fig = gcf()
# figure.axis('on')

x, y, z = np.mgrid[0:x_size, 0:y_size, 0:z_size]

v = np.zeros((50,50,50))

for k in range(50):
    for j in range(50):
        for i in range(50):
            idx = k * y_size * x_size + j * x_size + i
            v[i][j][k] = data[idx]


v = data.reshape((50, 50, 50))
vol = volume_slice(x, y, z, v, plane_orientation='z_axes')
