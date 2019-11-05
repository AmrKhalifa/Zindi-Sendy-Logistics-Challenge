# Zindi-s-Sendy-Logistics-Challenge

### How to use DataProcessor from data_processor.py module 
#### Step 1: instantiate instance from DataProcessor: 
  you have to give:
  
                   1.(str) the path to the file. 
                   2.(bool) to indicate if the this is the test data or not. 
                   3.(bool) to indicate either to drop the columns in the train file that are not found in the test file or not. 
                   
             
#### Step 2: call the method get_numpy_data() from the instance instantiated in step 1 above: 
**You should provide as parameters**: 

                  1.fillna: (Bool) if True it fills the missing columns in the dataset with the mean values for that column.
                  2.encode: (Bool)  if True it encodes the categorical variables with integer values.
                  3.np_split: (Bool) if True it splits to train and validation sets and returns 4 numpy arrays.
                  4.encode_user: (Bool) if True it encodes the user using 1-hot encoding (not implemented yet): default is False.
                  5.normalize: (Bool) if True it normalizes the dataset with z-score i.e. for each column it subtracts the mean value and divides by the standard deviation. 
                 
                  
**It returns**: 

        if np_split is True it returns 4 numpy arrays (x_train, x_valid, y_train, y_valid)
        if np_split is false it returns 2 numpu arrays (xtrain, y_train)
	                



### example
```
from data_processor import DataProcessor
processor = DataProcessor("../data/Train.csv", test = False, minimal = True)
x_train, x_valid, y_train, y_valid = processor.get_numpy_data(True, True, True, False, True)
```

