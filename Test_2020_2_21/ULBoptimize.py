import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import ast
import statistics
from os import listdir
from scipy.optimize import minimize

seed = 7
np.random.seed(seed)
    
SAVE_FIG=False
SAVE_DATA=True
FOLD_NUM=5
tested_noise=[]
new_tested_noise=[]
best_lambda=[]
new_best_lambda=[]
punish_set=range(16)
        
def Kron(x1, x2):
    if x1==x2:
        return 1
    else:
        return 0
    
def k(x1, x2, θ):
    L, σf, σn =  θ # can be adjusted
    return σf**2 * np.e**(-(x1-x2)**2 / 2*L**2) + σn**2 * Kron(x1, x2)
 
# optimize through Bayesian method
def optimize(x, y, θ, location, saveFig=True):
    samples=np.linspace(min(x)-0.2*(max(x)-min(x)), max(x)+0.2*(max(x)-min(x)), 2000)
    x_pred, y_pred, y_ub, y_lb=[], [], [], []
    for xs in samples:
        mat=[]
        mat2=[]
        for i in range(len(x)):
            row=[]
            for j in range(len(x)):
                row.append(k(x[i], x[j], θ))
            mat2.append(k(xs, x[i], θ))
            mat.append(row)
            
        K=np.matrix(mat)
        Ks=np.matrix(mat2)
        Kss=k(xs, xs, θ)
        
        y_mean=np.matmul(np.matmul(Ks, K.I), y)[0, 0]
        y_var= Kss-np.matmul(np.matmul(Ks, K.I), Ks.T)[0, 0]
        if y_var<0:
            print(y_var)
            y_var=0
        x_pred.append(xs)
        y_pred.append(y_mean) 
        y_ub.append(y_mean + 1.96*np.sqrt(y_var))
        y_lb.append(y_mean - 1.96*np.sqrt(y_var))
        
    ax = plt.subplot()
    ax.plot(x_pred, y_pred, c='b')
    ax.plot(x_pred, y_ub, c='c')
    ax.plot(x_pred, y_lb, c='c')
    ax.plot(x, y,'.', c='r')
    ax.set(title='Prediction')
    if saveFig:
        plt.savefig(location+'optimize.png')
    return x_pred, y_pred, y_ub, y_lb

# optimize the θ parameter
def paramOptimize(x, y):
    def paramOptimizer(θ):
        samples=np.linspace(min(x)-0.2*(max(x)-min(x)), max(x)+0.2*(max(x)-min(x)), 2000)
        x_pred, y_pred, y_ub, y_lb=[], [], [], []
        for xs in samples:
            mat=[]
            mat2=[]
            for i in range(len(x)):
                row=[]
                for j in range(len(x)):
                    row.append(k(x[i], x[j], θ))
                mat2.append(k(xs, x[i], θ))
                mat.append(row)
                
            K=np.matrix(mat)
            Ks=np.matrix(mat2)
            Kss=k(xs, xs, θ)
            
            y_mean=np.matmul(np.matmul(Ks, K.I), y)[0, 0]
            y_var= Kss-np.matmul(np.matmul(Ks, K.I), Ks.T)[0, 0]
            if y_var<0:
                print(y_var)
                y_var=0
            x_pred.append(xs)
            y_pred.append(y_mean) 
            y_ub.append(y_mean + 1.96*np.sqrt(y_var))
            y_lb.append(y_mean - 1.96*np.sqrt(y_var))
                    
        eva =(0.5 * np.matmul(np.matmul(y.T, K.I), y) + 0.5 * np.log(np.linalg.det(K)) + len(K)/2 * np.log(2*np.pi))[0,0]
        print(θ, eva)
        return eva
    return paramOptimizer


def solveParam(x, y, x0= [0.1, 10, 1]):
    x1=np.array(x)
    y1=np.array(y)
    res = minimize(paramOptimize(x1, y1), x0, method='Nelder-Mead')
    return res.x
    #[ 0.02221847 10.10177288  0.65953988] 19.156197098280682
    
    
