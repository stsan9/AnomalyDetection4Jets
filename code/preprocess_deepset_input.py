#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
import math
import h5py


# In[2]:


# m_12 = sqrt ( (E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2 )
def invariant_mass(jet1, jet2):
    return math.sqrt((jet1.e + jet2.e)**2 - (jet1.px + jet2.px)**2 - (jet1.py + jet2.py)**2 - (jet1.pz + jet2.pz)**2)


# In[3]:


path = '../events_LHCO2020_BlackBox1.h5'


# In[4]:


chunk_size = 1000000
total_size = 1000000 # 1 mil max

def generator(path, chunk_size=10000,total_size=1000000):
    i = 0
    
    while True:
        yield pd.read_hdf(path,start=i*chunk_size, stop=(i+1)*chunk_size)
        
        i+=1
        if (i+1)*chunk_size > total_size:
            i=0

gen = generator(path, chunk_size, total_size)


# In[5]:

# extract data for scaling
data = []

for iteration in range(total_size // chunk_size):
    
    events = np.array(next(gen))
    rows = events.shape[0]
    cols = events.shape[1]

    for i in range(rows):
        for j in range(100):
            particle = [events[i][j*3], events[i][j*3+1], events[i][j*3+2]]
            data.append(particle)

print("Finished array construction")
data = np.array(data)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)

# In[6]:

data = data.reshape((total_size,100,3))


# In[9]:

with h5py.File('preprocessed_goods/bb1_100x3.h5','w') as f:
    f.create_dataset('df', shape=(total_size,100,3), data=data, dtype='float64')
