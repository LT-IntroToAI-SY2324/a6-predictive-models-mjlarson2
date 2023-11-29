import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x, y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
rSquared = model.score(x, y)

# Print out the linear equation and r squared value
print(f'y = {coef}x + {intercept}')
print(f"R^2: {rSquared}")
# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
print(f"The blood pressure of someone who is 43 years old is predicted to be {model.predict([[43]])}")
# Create the model in matplotlib and include the line of best fit
plt.figure(figsize=(5,4))
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x, y)
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")
plt.show()