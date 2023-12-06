import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles(000)","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
print(f'Coefficients: {np.around(model.coef_, 2)}')
print(f'Bias: {round(float(model.intercept_), 2)}')
print(f'R squared: {round(model.score(x, y), 2)}')
predict = np.around(model.predict(xtest))

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
for i in range(len(xtest)):
    actual = ytest[i]
    predictedy = predict[i]
    xcoord = xtest[i]
    print(f"Miles(000): {xcoord[0]} Age: {xcoord[1]} Actual: {actual} Predicted: {predictedy}")


# For the writeup - changes every time its run b/c of train-test-split
print(np.around(model.predict([[89, 10]])))
print(np.around(model.predict([[150, 20]])))