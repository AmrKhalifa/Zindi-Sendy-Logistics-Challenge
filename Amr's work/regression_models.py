
# coding: utf-8

# In[9]:

from data_processor import DataProcessor
import numpy as np
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 


# In[2]:

def evaluate(model, x_valid, y_valid):
    error = mean_squared_error(y_valid.ravel(), model.predict(x_valid))
    var = np.var(y_valid.ravel())
    #print("The MSE error is: ", error)
    #print("The variance of the validation set is: ", var)
    r_2 = 1 - error / var

    #print("The model explians" ,r_2 ," of the variance in data")
    return r_2
    


# In[5]:

file_1 = "../data/Train.csv"
file_2 = "../data/additional_data/trainRoot_edited.csv"

processor = DataProcessor(file_1, file_2, test = False, minimal = True)
x_train, x_valid, y_train, y_valid = processor.get_numpy_data(fillna = True, additional = True,
                                                                            encode = True, np_split = True, enocde_user = False,
                                                                            normalize = True, drop_ones = True)


# In[4]:

results = []


# In[7]:

# i =0 
# for depth in range (1, 10, 2): 
#     for estimators in range(100, 1500, 300):
#         for weight in range(1, 50, 3):
#             i += 1
#             param_dist = {'objective':'reg:squarederror', 'n_estimators':estimators, 'max_depth':depth, 'min_child_weight': weight}
#             bst = xgb.XGBRFRegressor(**param_dist)
#             #print("=*="*20)
#             bst.fit(x_train, y_train.ravel(), eval_set=[(x_valid, y_valid)], verbose = True)
#             #print("depth = ",depth, "estimators = ", estimators )
#             #results = bst.evals_result()
#             r_2 = evaluate(bst, x_valid, y_valid)
#             results.append((r_2, depth, estimators, weight))
            
#             if i % 5 ==0:
#                 with open ("results.txt", "w") as f:
#                     for element in results[i-5:i]:
#                         f.write(str(element)+'\n')
# print(results)


# # In[8]:

# evaluate(bst, x_valid, y_valid)


# In[9]:

#for result in results:
    #print(result)


# In[10]:

# rs = []
# ests = []
# depths = []
# ws = []
# for result in results:
#     rs.append(result[0])
#     depths.append(result[1])
#     ests.append(result[2])
#     ws.append(result[3])


# In[11]:

#plt.plot(ests, rs)
#plt.plot(depths, rs)
#plt.plot(ws, rs)
# plt.plot(rs)
# plt.show()


# In[12]:




# ### Predictions 

# In[10]:

file_1 = "../data/Test.csv"
file_2 = "../data/additional_data/testRoot_edited.csv"

processor = DataProcessor(file_1, file_2, test = True, minimal = True)
x_test = processor.get_numpy_data(fillna = True, additional = True,
                                  encode = True, np_split = False, enocde_user = False,
                                  normalize = True, drop_ones = False)

#print(x_test.head())
# In[6]:

param_dist = {'objective':'reg:squarederror', 'n_estimators':1300, 'max_depth':9, 'min_child_weight': 49}
bst = xgb.XGBRFRegressor(**param_dist)
bst.fit(x_train, y_train.ravel(), eval_set=[(x_valid, y_valid)], verbose = True)

pr = bst.predict(x_test)
            
print(pr)

# In[ ]:

#test

