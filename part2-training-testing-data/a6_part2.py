import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1,1)

# Create the model
model = LinearRegression().fit(xtrain, ytrain)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
bias = round(float(model.intercept_), 2)
rSquared = model.score(xtrain, ytrain)

# Print out the linear equation and r squared value:
print(f"y = {coef}x + {bias}")
print(f"r squared: {rSquared}")
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1, 1)
# get the predicted y values for the xtest values - returns an array of the results
predictions = model.predict(xtest)
# round the value in the np array to 2 decimal places
predictions = np.around(predictions)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for i in range(len(xtest)):
    actual = ytest[i] 
    predictedY = predictions[i] 
    xCoord = xtest[i] 
    print("@ x = ", float(xCoord), "Predicted y:", predictedY, "Actual y:", actual)


'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.plot(5, 4)
plt.scatter(xtrain, ytrain, c="blue", label='Training Data')
plt.scatter(xtest, ytest, c="purple", label="Testing Data")
plt.scatter(xtest, predictions, c="red", label="Predictions")
plt.plot()
plt.xlabel("Temperature ºF")
plt.ylabel("Chirps per Minute")
plt.title("Cricket Chirps by Temperature")
plt.plot(x, coef*x + bias, c="r", label="Line of Best Fit")

plt.legend()
plt.show()