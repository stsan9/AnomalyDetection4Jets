#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
from sklearn import preprocessing
from scipy.stats import iqr
import tensorflow as tf
import math
import h5py


# # Pre processing

# In[2]:


path = '/anomalyvol/data/events_LHCO2020_BlackBox1.h5'


# In[3]:


chunk_size = 100000
total_size = 1000000 # 1 mil max

def generator(path, chunk_size=10000,total_size=1000000):
    i = 0
    
    while True:
        yield pd.read_hdf(path,start=i*chunk_size, stop=(i+1)*chunk_size)
        
        i+=1
        if (i+1)*chunk_size > total_size:
            i=0

gen = generator(path, chunk_size, total_size)


# In[ ]:


data = []

for iteration in range(total_size // chunk_size):
    
    events = np.array(next(gen))
    rows = events.shape[0]
    cols = events.shape[1]

    for i in range(rows):
        pseudojets_input = np.zeros(len([x for x in events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
        for j in range(cols // 3):
            if (events[i][j*3]>0):
                pseudojets_input[j]['pT'] = events[i][j*3]
                pseudojets_input[j]['eta'] = events[i][j*3+1]
                pseudojets_input[j]['phi'] = events[i][j*3+2]
            pass
        # cluster jets from the particles in one observation
        sequence = cluster(pseudojets_input, R=1.0, p=-1)
        jets = sequence.inclusive_jets()
        for k in range(len(jets)): # for each jet get (px, py, pz, e)
            jet = []
            jet.append(jets[k].px)
            jet.append(jets[k].py)
            jet.append(jets[k].pz)
            jet.append(jets[k].e)
            jet.append(jets[k].pt)
            jet.append(jets[k].eta)
            jet.append(jets[k].phi)
            jet.append(jets[k].mass)
            jet.append(iteration * chunk_size + i)  # event index
            data.append(jet)


# In[ ]:


loaded_data = data
# data = loaded_data


# In[ ]:


data = pd.DataFrame(data)
data.columns = ['px','py','pz','e','pt','eta','phi','mass','event']


# In[ ]:


data.to_hdf('/anomalyvol/data/jet_ver/bb1_ae_input.h5',key='df',mode='w')

