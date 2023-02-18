import sys
import os
from itertools import chain 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class tpred(object):
    def __init__(self, training_set, lastScore, score, steps ):
        self.training_set = training_set,
        self.lastPoint = lastScore,
        self.score = score,
        self.steps = steps
        self.sc = MinMaxScaler(feature_range = (0,1))

    def difa(a,b):
        diff = (a-b)/b
        if(diff > 0):
            return diff
        else:
            return 0

    def normalize(p, n=10, m=100):
        v = p
        nm1 = (n-1)
        v = np.round(v*m) + np.round((n-1)/2)
        if v > nm1: 
            return nm1
        elif v < 0: 
            return 0 
        else: 
            return v
    
    def getSequence(self, data):
        sequence = tf.Variable(tf.size(data) - 1)
        print("check sequence size: ", sequence)
        for i in range(0, 20):
            diff_a = self.difa(data[i], data[i+1])
            p = self.normalize(diff_a)
            #print(int(p))
            data_scaled = tf.Variable(self.sc.fit_transform(self.data))
            data_scaled[i].assign(int(p))
        print(data_scaled)
        for i in range(0, 19):
            for x in range(i, i+1):
                d1 = data_scaled[tf.size(data_scaled) - (tf.size(data_scaled) - x)]
                d2 = data_scaled[x + 1]
                if(d2 - d1 > 0):
                    sequence[i].assign(2)
                else:
                    sequence[i].assign(2)
        
    def preEvaluate(self):
        main_cv = (tf.math.reduce_std(self.training_set)/tf.math.reduce_mean(self.training_set))*100
        main_sequence = self.sequence(self.training_set)
        v = tf.Variable(tf.random.shuffle(self.training_set))
        v_sequence = self.sequence(v)
        cv = (tf.math.reduce_std(v)/tf.math.reduce_mean(v))*100
        return cv

    def calcVariation(x):

