import pandas as pd
import collections
import re
import pprint as pp
import numpy as np

from sklearn.metrics import precision_recall_curve, auc, log_loss
from sklearn.preprocessing import StandardScaler

import math
import gzip
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from datetime import datetime
import random
random.seed(0)
import os

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import train_test_split 

import gensim

from keras import Sequential
from keras.layers import Dense,Dropout,BatchNormalization, LeakyReLU
from keras import optimizers
from keras.callbacks import EarlyStopping
from imblearn.under_sampling import RandomUnderSampler
from tqdm.notebook import tqdm
import batch_processing


    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0



