#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:04:13 2018

@author: vorstenbosch
"""

import pandas as pd
import numpy as np

info = np.loadtxt('census-income.names', dtype=str, delimiter = '\n', skiprows=142)

# Loads the names from the .names file
names = []
for i in info:
    split = i.split(":")
    if split[1] != ' ignore.':
        names.append(split[0])
    
data = pd.read_table("census-income.data", index_col=False,  delimiter =',', names=names, header=None)

data_pd = data
data = np.array(data)
ncols = np.shape(data)[1]

uniques = np.zeros(ncols ,dtype=int)

#Assignment 1a
f = open("1D_length.csv", 'w+')
f.write("col name,length\n")
values = []
for i in range(ncols):
    values.append(np.unique(data[:,i]))
    uniques[i] = int(len(values[i]))
    f.write(str(names[i]) + "," + str(uniques[i]) + "\n")
f.close()

#Assignment 1b
target = uniques.argsort()[::-1][0]
largest = uniques.argsort()[::-1][1]
smallest = uniques.argsort()[0]

apex = np.sum(data[:,target])
f = open("0D.csv", 'w+')
f.write(str(apex))
f.close()

f = open("1D_largest.csv", 'w+')
f.write(str(names[largest]) + ",measure" + "\n" )
for v in values[largest]:
    indices = np.where(data[:,largest] == v)
    f.write(str(v) + "," + str(np.sum(data[indices,target])) + "\n")
f.close()

f = open("1D_smallest.csv", 'w+')
f.write(str(names[smallest]) + ",measure" + "\n" )
for v in values[smallest]:
    indices = np.where(data[:,smallest] == v)
    f.write(str(v) + "," + str(np.sum(data[indices,target])) + "\n")
f.close()

for col, vals in enumerate(values):
    print(col)
    f = open("2D_%s.csv"%names[col], 'w+')
    f.write("target," + str(names[col]) + ",measure" + "\n" )
    for v_t in values[target]:
        for v_i in vals:
            indices = np.where((data[:,col] == v_i) & (data[:,target]== v_t))
            if len(indices[0]) != 0:
                f.write(str(v_t) + "," + str(v_i) + "," + str(np.sum(data[indices,target])))
    f.close()

