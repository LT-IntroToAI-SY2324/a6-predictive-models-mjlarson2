import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import numpy as np

ref = ['Yes', 'No']

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
# print(x)
# print(y)
# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler().fit(x)
# Step 3: Transform the data
x = scaler.transform(x)
# Step 4: Split the data into training and testing data
xtrain, xtest, ytrain, ytest = tts(x, y)
# Step 5: Fit the data
# Step 6: Create a LogsiticRegression object and fit the data
model = LR().fit(xtrain, ytrain)
# Step 7: Print the score to see the accuracy of the model
print(f'Accuracy: {model.score(xtest, ytest)}')
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
for i in range(0, len(xtest)):
    x = xtest[i]
    x = x.reshape(-1, 3)
    p = int(model.predict(x))
    y = ytest[i]
    print(x)
    print(f'Predicted : {ref[p]}; Actual: {ref[y]}')
print(model.predict(np.asarray([34, 56000, 1]).reshape(1, -1)))
    
    