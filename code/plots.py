import pandas as pd
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
from sklearn import preprocessing
from scipy.stats import iqr
import tensorflow as tf
import math
import keras
from keras import metrics, losses
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def get_data():
    pass

def genplots(modelname):
    model = keras.models.load_model(modelname)

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, help="path to the model", required=True)
    
    genplots(modelname)