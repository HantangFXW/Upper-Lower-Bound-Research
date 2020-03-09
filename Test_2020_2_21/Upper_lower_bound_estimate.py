import numpy as np
from ULBfunctions import test
from ULBoptimize import optimize
from ULBoptimize import solveParam
import csv

seed = 7
np.random.seed(seed)
T_set=[1,5, 10]
test_range=range(5,50,5)
testName="hetero_model 3"


for T in T_set:
    location = testName + "/" +str(T)+"/" 
    lambda_range=list(test_range)
    CWC=test(T, lambda_range, location, True)
       
    with open((location + "data0.csv"), 'w',  newline='') as f:
        wr=csv.writer(f)
        wr.writerows([CWC])
        wr.writerows([lambda_range])

    
    for i in range(1,3): 
        [x_pred, y_pred, y_ub, y_lb]=optimize(lambda_range, CWC, [0.0222, 10.1017, 0.6595], location)
        
        x_min = float(x_pred[y_pred.index(min(y_pred))])
        distance = (lambda_range[1]-lambda_range[0]) / 4
        lambda_range = [x_min-2*distance, x_min-distance, x_min, x_min+distance, x_min+2*distance]
        CWC=test(T, lambda_range, location)
        
        with open((location + "data"+str(i)+".csv"), 'w',  newline='') as f:
            wr=csv.writer(f)
            wr.writerows([CWC])
            wr.writerows([lambda_range])                
    print(CWC)


print('Program end')