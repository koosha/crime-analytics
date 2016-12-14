
# coding: utf-8

# In[1]:

import numpy as np
import math
import time


def MLSSVRTrain(trnX, trnY, gamma, lambdaa, p):
    if (len(trnX) != len(trnY)):
        print('The number of rows in trnX and trnY must be equal.')
        return 0, 0
    
    l = np.shape(trnY)[0]
    m = np.shape(trnY)[1]
    if not gamma or  not lambdaa:
        gamma=1
        lambdaa = 1
    K = Kerfun('rbf', trnX, trnX, p, 0)
    H = np.tile(K, (m, m))+ np.identity(m * l) / gamma
    P = np.zeros((m*l,m))
    for  t in range(1,m+1):
        idx1 = l * (t - 1) + 1;
        idx2 = l * t;
        H[idx1-1:idx2, idx1-1:idx2] = H[idx1-1:idx2, idx1-1:idx2] + K*(m/lambdaa) 
        P[idx1-1:idx2 , t-1] = np.transpose(np.ones((l,1)))

    eta, _, _, _ = np.linalg.lstsq(H,P)
    ttrnY = np.transpose(trnY.ravel(order='F'))
    nu, _, _, _ = np.linalg.lstsq(H,ttrnY)
    S = np.dot(np.transpose(P),eta)
    b = np.dot(np.dot(np.linalg.inv(S), np.transpose(eta)), ttrnY)
    alpha = nu - np.dot(eta, b)
    alpha = np.reshape(alpha,(l,m), order='F')
    return alpha, b


# In[2]:

def Kerfun(kernel,X,Z,p1,p2):
	sx  = np.shape(X)
	sz  = np.shape(Z)
	if sx[1] != sz[1]:
		K = []
		print("The second dimensions for X and Z must be agree.")
		return K

	if kernel.lower() == 'linear':
		K = np.dot(X,np.transpose(Z))
	elif kernel.lower() == 'poly':
		K = np.power((np.dot(X,np.transpose(Z))+p1),p2)
	elif kernel.lower() == 'rbf':
		K = np.exp(-p1*(np.tile(np.sum(X*X, axis=1),(sz[0],1)).T+np.tile(np.transpose(np.sum(Z*Z, axis=1)),(sx[0],1))-2*X.dot(np.transpose(Z))))
	elif kernel.lower() == 'erbf':
		K = np.exp(-np.sqrt(np.tile(np.sum(X*X, axis=1),(sz[0],1)).T+np.tile(np.transpose(np.sum(Z*Z, axis=1)),(sx[0],1))-2*X.dot(np.transpose(Z)))/(2*p1**2))+p2
	elif kernel.lower() == 'sigmoid':
		K = np.tanh(p1*X.dot(np.transpose(Z))/sx[1]+p2)
	else:
		K = X.dot(np.transpose(Z))+p1+p2
	return K



# In[3]:

import numpy as np
import math




def MLSSVRPredict(tstX, tstY, trnX, alpha, b, lambdaa, p):
    if np.shape(tstY)[1] != len(b):
        print('The number of column in tstY and b must be equal.')
        return

    m = np.shape(tstY)[1]
    l = np.shape(trnX)[0]
    
    if (np.shape(alpha)[0] != l or np.shape(alpha)[1] != m):
        print('The size of alpha should be ' + l + '*' + m)
        return

    tstN = np.shape(tstX)[0];
    b = np.transpose(b.ravel(order='F'))
    K = Kerfun('rbf', tstX, trnX, p, 0)
    predictY = np.transpose(np.tile(np.transpose(np.sum(np.dot(K,alpha),axis=1)), (m,1 ))) + np.dot(K,alpha)*(m/lambdaa) + np.tile(np.transpose(b), (tstN, 1))


    TSE = np.zeros((1,m))
    R2 = np.zeros((1,m))
    for t in range(m):
        ppp = predictY[:, t]
        ttt = tstY[:, t]
   
        TSE[0,t] = np.sum((ppp-ttt)**2)
        R = np.corrcoef(ppp, tstY[:,t])
        if (len(R)>1):
            R2[0,t] = R[0,1]**2
    return predictY, TSE, R2


# In[4]:

import random
import numpy as np
import math
def GridMLSSVR(trnX,trnY,fold):
    gamma = np.power(np.arange(-5, 15, 5),2)
    lambdaa = np.power(np.arange(-10, 10, 5),2)
    p = np.power(np.arange(-15, 3, 5),2)
    tYs = np.shape(trnY)
    m = tYs[1]

    trnX,trnY = random_perm(trnX, trnY);

    MSE_best = math.inf

    MSE = np.zeros((fold, m))
    curR2 = np.zeros((1, m))
    R2 = np.zeros((1, m))

    sp = np.shape(p)
    sg = np.shape(gamma)
    sl = np.shape(lambdaa)
    for i in range(sg[0]):
        for j in range(sl[0]):
            for k in range(sp[0]):
                predictY = []
                for v in range(fold):
                    print(str(i)+' gamma '+str(j)+'  lambda  '+str(k)+' p ')
                    train_inst, train_lbl, test_inst, test_lbl = folding(trnX, trnY, fold, v)
                    
                    alpha, b = MLSSVRTrain(train_inst, train_lbl, gamma[i], lambdaa[j], p[k])
                    tmpY, ms, xx = MLSSVRPredict(test_inst, test_lbl,train_inst,alpha,b,lambdaa[j],p[k])
                    MSE[v,:] = ms

                    if not np.shape(predictY)[0]:
                        predictY = tmpY
                    else:
                        predictY = np.concatenate((predictY, tmpY), axis=0)

                sy = np.shape(trnY)
                curMSE = np.sum(MSE)/ (sy[0]*sy[1])

                if MSE_best > curMSE:
                    gamma_best = gamma[i]
                    lambda_best = lambdaa[j]
                    p_best = p[k]
                    MSE_best = curMSE
    return gamma_best, lambda_best, p_best, MSE_best

                    



def random_perm(svm_inst, svm_lbl):
    
    random.seed(a=1, version=2)
    n = np.shape(svm_inst)[0]
    for i in range(n):
        k = round(i + (n - i) * random.random())-1
        svm_inst[[k, i], :] = svm_inst[[i, k], :]
        svm_lbl[[k, i], :] = svm_lbl[[i, k], :]
    return svm_inst, svm_lbl




def folding(svm_inst, svm_lbl, fold, k):
    n = np.shape(svm_inst)[0]
    start_index = round((k - 1)*n/fold) + 1;
    end_index = round(k*n/fold);
    test_index = [start_index, end_index]

    test_inst = svm_inst[test_index, :];
    test_lbl = svm_lbl[test_index, :];

    train_inst = svm_inst;
    #train_inst[test_index, :] = [];
    np.delete(train_inst, test_index, axis=0)
    train_lbl = svm_lbl;
    #train_lbl[test_index, :] = [];
    np.delete(train_lbl, test_index, axis=0)

    return train_inst, train_lbl, test_inst, test_lbl


# In[5]:

# import pandas as pd
# df = pd.read_csv('../data/test1.csv')


# In[6]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import preprocessing, linear_model
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import ShuffleSplit
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
# def train_test_spliter(df, random_seed=0, test_size=0.20, n_splits=1):
#     trainm = preprocessing.maxabs_scale(df, axis=0, copy=True)
#     col = df.columns
#     df1 = pd.DataFrame(data=trainm, columns=col)
#     rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
#     for train_index, test_index in rs.split(df):
#         pass
#     traindf = pd.DataFrame(data=df, index=train_index)
#     testdf = pd.DataFrame(data=df, index=test_index)
#     return traindf, testdf


# In[7]:

# traindf, testdf = train_test_spliter(df)


# In[8]:

# train_m = traindf.values


# In[9]:

# test_m = testdf.values


# In[10]:

# train_instance =train_m[:,0:3]


# In[11]:

# train_label = train_m[:,3:5]


# In[12]:

# test_instance =test_m[:,0:3]


# In[13]:

# test_label = test_m[:,3:5]


# In[14]:

# X = train_instance
# Z = train_instance


# In[15]:

# K = np.exp(-1*(np.tile(np.sum(X*X, axis=1),(sz[0],1)).T+np.tile(np.transpose(np.sum(Z*Z, axis=1)),(sx[0],1))-2*X.dot(np.transpose(Z))))


# In[16]:

# import scipy.io as sio
# sio.savemat('trnX.mat',{'Matrix1':train_instance})
# sio.savemat('trnY.mat',{'Matrix1':train_label})
# sio.savemat('tstX.mat',{'Matrix1':test_instance})
# sio.savemat('tstY.mat',{'Matrix1':test_label})


# In[17]:

# alpha,b = MLSSVRTrain(train_instance,train_label,1,1,1)


# In[18]:

# MLSSVRPredict(test_instance, test_label, train_instance, alpha, b, 1, 1)


# In[19]:

# trnX = train_instance
# trnY = train_label
# gamma_best, lambda_best, p_best, MSE_best = GridMLSSVR(trnX,trnY,5)


# In[20]:

# alpha,b = MLSSVRTrain(train_instance,train_label,gamma_best,lambda_best, p_best)


# In[21]:

# MLSSVRPredict(test_instance, test_label, train_instance, alpha, b, lambda_best, p_best)


# In[22]:

# a = [[ 0.09775068,  0.58191474],
#         [ 0.85833902,  2.02129018],
#         [ 0.71235429,  1.0754867 ],
#         [ 0.30791839,  2.47933833],
#         [ 0.27987689,  1.69218858],
#         [ 0.96028704,  1.63338285],
#         [ 0.32607547,  1.7290909 ],
#         [ 0.56928159,  1.62021161],
#         [ 1.16633211,  1.4346901 ],
#         [ 0.46158372,  0.73371174],
#         [ 0.03763364,  0.19991961],
#         [ 0.49851494,  1.53597745],
#         [ 0.13167075,  0.7593729 ],
#         [ 0.43049893,  1.13138277],
#         [ 0.78857616,  1.55285531],
#         [ 0.56473995,  0.636297  ],
#         [ 0.68558722,  2.21831356],
#         [ 0.54325872,  2.66079606],
#         [ 0.76211686,  1.95544598],
#         [ 0.80837568,  1.53794821],
#         [ 0.55564975,  2.26977522],
#         [ 1.05892341,  2.25964809],
#         [ 0.35744775,  1.74742411],
#         [ 0.29845019,  1.53618925],
#         [ 1.40612547,  2.19739105],
#         [ 0.49486454,  2.04309745],
#         [ 0.62235759,  1.91928615],
#         [ 0.38614102,  0.1327358 ],
#         [ 1.09180796,  1.47637161],
#         [ 0.94800672,  1.13360841],
#         [ 0.32403001,  0.93292027],
#         [ 0.42731018,  0.21242981],
#         [ 0.37702869,  0.70969398],
#         [ 0.08952517,  1.43678308],
#         [ 0.30913086,  0.60977842],
#         [ 0.73596734,  1.65786412],
#         [ 0.1425106 ,  2.68264293],
#         [-0.00449228,  1.23209009],
#         [ 0.46487643,  0.18806433],
#         [ 0.26407622,  0.54927083]]


# In[23]:

# from sklearn.metrics import mean_squared_error

# mean_squared_error(a, test_label)


# In[24]:

import pandas as pd
import seaborn as sns
import numpy as np


from sklearn import linear_model, svm, tree
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
df = pd.read_csv('../merged_data/ready.csv')
df = df.fillna(0)


# In[25]:

df.columns
df = df[['Canadian Citizen', 'No Response(Citizen)', 'Non-Canadian Citizen',
       'Refugee', 'No Longer In Use', 'Occupied', 'Unoccupied',
       'Employed 0-30 Hours', 'Employed 30+ Hours', 'Gr.10 - Gr.12',
       'Gr.7 - Gr.9', 'Homemaker', 'Kindergarten - Gr.6',
       'No Response(Employment)', 'Permanently Unable to Work',
       'Post Secondary Student', 'Preschool', 'Retired', 'Unemployed',
       'Common Law', 'Married', 'Never Married', 'No Response(Marital)',
       'Separated/Divorced', 'Widowed', 'Bicycle',
       'Car/Truck/Van (as Driver)', 'Car/Truck/Van (as Passenger)',
       'No Response(Transportation)', 'Other', 'Public Transit', 'Walk',
       'Catholic', 'No Response(School)', 'Public', 'Assault', 'Break and Enter', 'Homicide',
       'Robbery', 'Sexual Assaults', 'Theft From Vehicle', 'Theft Of Vehicle',
       'Theft Over $5000']]


# In[26]:

def train_test_spliter(df, random_seed=0, test_size=0.20, n_splits=1):
    # trainm = preprocessing.maxabs_scale(df, axis=0, copy=True)
    col = df.columns
    df1 = pd.DataFrame(data=df, columns=col)
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
    for train_index, test_index in rs.split(df):
        pass
    traindf = pd.DataFrame(data=df, index=train_index)
    testdf = pd.DataFrame(data=df, index=test_index)
    return traindf, testdf


traindf, testdf = train_test_spliter(df)

X_train = pd.DataFrame(data=traindf, columns=['Canadian Citizen', 'No Response(Citizen)', 'Non-Canadian Citizen',
       'Refugee', 'No Longer In Use', 'Occupied', 'Unoccupied',
       'Employed 0-30 Hours', 'Employed 30+ Hours', 'Gr.10 - Gr.12',
       'Gr.7 - Gr.9', 'Homemaker', 'Kindergarten - Gr.6',
       'No Response(Employment)', 'Permanently Unable to Work',
       'Post Secondary Student', 'Preschool', 'Retired', 'Unemployed',
       'Common Law', 'Married', 'Never Married', 'No Response(Marital)',
       'Separated/Divorced', 'Widowed', 'Bicycle',
       'Car/Truck/Van (as Driver)', 'Car/Truck/Van (as Passenger)',
       'No Response(Transportation)', 'Other', 'Public Transit', 'Walk',
       'Catholic', 'No Response(School)', 'Public'])
X_train = preprocessing.maxabs_scale(X_train, axis=0, copy=True)

y_train = pd.DataFrame(data=traindf, columns=['Assault', 'Break and Enter', 'Homicide',
       'Robbery', 'Sexual Assaults', 'Theft From Vehicle', 'Theft Of Vehicle',
       'Theft Over $5000'])
X_test = pd.DataFrame(data=testdf, columns=['Canadian Citizen', 'No Response(Citizen)', 'Non-Canadian Citizen',
       'Refugee', 'No Longer In Use', 'Occupied', 'Unoccupied',
       'Employed 0-30 Hours', 'Employed 30+ Hours', 'Gr.10 - Gr.12',
       'Gr.7 - Gr.9', 'Homemaker', 'Kindergarten - Gr.6',
       'No Response(Employment)', 'Permanently Unable to Work',
       'Post Secondary Student', 'Preschool', 'Retired', 'Unemployed',
       'Common Law', 'Married', 'Never Married', 'No Response(Marital)',
       'Separated/Divorced', 'Widowed', 'Bicycle',
       'Car/Truck/Van (as Driver)', 'Car/Truck/Van (as Passenger)',
       'No Response(Transportation)', 'Other', 'Public Transit', 'Walk',
       'Catholic', 'No Response(School)', 'Public'])
X_test = preprocessing.maxabs_scale(X_test, axis=0, copy=True)

y_test = pd.DataFrame(data=testdf, columns=['Assault', 'Break and Enter', 'Homicide',
       'Robbery', 'Sexual Assaults', 'Theft From Vehicle', 'Theft Of Vehicle',
       'Theft Over $5000'])


# In[27]:

#GridMLSSVR(X_train,y_train.values,1)


# In[ ]:

xtr = X_train[0:1000]
ytr = y_train.values[0:1000]
GridMLSSVR(xtr,ytr,1)


# In[34]:




# In[ ]:



