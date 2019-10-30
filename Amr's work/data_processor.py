import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("data/Train.csv")

date_attributes = ['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 'Pickup - Time', 'Arrival at Destination - Time']	

train_df = train_df.drop(['Order No'], axis = 1)
train_df = train_df.drop(['Precipitation in millimeters'], axis = 1)

train_df = train_df.fillna(train_df.mean())

for date in date_attributes: 
	train_df[date] = pd.to_datetime(train_df[date]).dt.strftime('%H:%M:%S')


categorical_attributes = ['User Id' , 'Vehicle Type', 'Platform Type', 'Rider Id', 'Personal or Business']

encoder = LabelEncoder()

encoded = train_df[categorical_attributes].apply(encoder.fit_transform)

train_df [categorical_attributes] = encoded [categorical_attributes]

feature_cols = train_df.columns.drop(['Time from Pickup to Arrival'])

X = train_df[feature_cols]
Y = train_df[['Time from Pickup to Arrival']]

x_train, x_valid, y_train, y_valid = train_test_split(X.values, Y.values, test_size=0.33)

#print(x_train)
#print(y_train)
print(" ")
print("The Shapes are: ")
print(x_train.shape)
print(y_train.shape)

print(type(x_train[0]))
print(x_train[0])
def main():
    pass

if __name__ == "__main__":
	main()