import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import csv


def functionF(start, end, samples, T):
    x = np.linspace(start, end, samples)
    g = np.square(x)+np.sin(x)+2
    y = g +np.random.normal(loc = 0, scale = g/T, size = x.size)
    return x, y

def compare(larger, smaller, punish_val):
    diff=smaller-larger
    sum=tf.reduce_mean(tf.square(diff))
    punish = tf.map_fn(lambda x: punish_val * tf.keras.backend.maximum(x, 0), diff)
    sum += tf.reduce_mean(tf.square(punish))
    return sum

def custom_loss_factory(punish):
    def custom_loss(y_true, y_pred):
        upper_b = y_pred[:,0]
        lower_b = y_pred[:,1]
        desired = y_true[:,0]
        return (compare(upper_b, desired, punish)+compare(desired, lower_b, punish))/ (tf.reduce_max(y_true)-tf.reduce_min(y_true))
    return custom_loss

def create_model(punish):    
    activator = 'elu'
    model=tf.keras.Sequential()
    model.add(layers.Dense(1, input_dim=1, kernel_initializer='normal', activation=activator))
    model.add(layers.Dense(32, activation=activator))
    model.add(layers.Dense(16, activation=activator))
    model.add(layers.Dense(2, activation='linear'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss=custom_loss_factory(punish),
                  metrics=['mae'])
    return model
def NMPIW(ub, lb, y_true):
    return np.mean(ub-lb) / (np.max(y_true)-np.min(y_true))
    
def PICP(ub, lb, y_true):
    size=np.size(ub)
    PICP=size
    for i in range(size):
        if(y_true[i] > ub[i] or y_true[i] < lb[i]):
            PICP -= 1
    return PICP/size
     
def CWC(ub, lb, y_true):
    μ=0.95
    η=100
    nmpiw=NMPIW(ub, lb, y_true)
    picp=PICP(ub,lb, y_true)
    if(picp>=μ):
        γ=0
    else:
        γ=1
    return (nmpiw * 100, picp * 100, nmpiw * (1 + γ * np.power(np.e, (-η*(picp-μ)))) * 100)
    
    
def test(T, lambdaRange, location, newModel=True, saveModel=True):     
    x, y= functionF(-10, 10, 2000, T)
    
    FOLD_NUM=5
    kfold = KFold(FOLD_NUM, True, 1)
    data_collection=[]
    #loop for different punish_val
    for i in lambdaRange:
        data_collection_sub=[]
        punish=i
        count=0
        for train, test in kfold.split(x):
            count+=1
            if newModel:
                model=create_model(punish)
                model.fit(x[train], y[train], epochs=1000, batch_size=32)
            else:
                model=tf.keras.models.load_model(location+'Kfolder_lambda_'+str(i)+'_'+str(count)+'.h5')
            
            test_result = model.predict(x[test], batch_size=32).astype(np.float64)
            test_ub = test_result[:,0]
            test_lb = test_result[:,1]
            
            
            directory1 = location + 'Kfolder_lambda_'+str(i)+'_'+str(count)+'.png'
            directory2 = location + 'Kfolder_lambda_'+str(i)+'_'+str(count)+'.h5'
            
            (a, b, c)=CWC(test_ub, test_lb, y[test])
            print('NMPIW:',a, 'PICP:', b, 'CWC:', c)
            data_collection_sub.append([a, b, c])
            print('Penalty_val', i, count, 'test continue')
            
            if saveModel:
                model.save(directory2)
            #reload models
            #new_model = tf.keras.models.load_model('my_model.h5')
            #new_model.summary()
            
        data_collection.append(data_collection_sub)
        print('Penalty', i, count, 'test finish')
        
    aver_CWC=[]
    aver_NMPIW=[]
    aver_PICP=[]
    count=0
    for i in lambdaRange:
        nmpiw=0
        picp=0
        cwc=0
        for j in range(FOLD_NUM):
            nmpiw += data_collection[count][j][0]
            picp += data_collection[count][j][1]
            cwc += data_collection[count][j][2]
        aver_NMPIW.append(nmpiw / FOLD_NUM)
        aver_PICP.append(picp / FOLD_NUM)
        aver_CWC.append(cwc / FOLD_NUM)
        count+=1;
            
    for j in range(FOLD_NUM):
        nmpiw_sub=[]
        picp_sub=[]
        cwc_sub=[]
        count=0
        for i in lambdaRange:
            nmpiw_sub.append(data_collection[count][j][0])
            picp_sub.append(data_collection[count][j][1])
            cwc_sub.append(data_collection[count][j][2])
            count+=1;        
    
    if newModel:
        with open((location + 'data.csv'), 'w',  newline='') as f:
            wr=csv.writer(f)
            wr.writerows(data_collection)
    return aver_CWC
