# Part 3 - Multivariable Linear Regression Writeup

After completing `a6_part3.py` answer the following questions

## Questions to answer

1. What is the R Squared coefficient for your model? What does that mean about the model in relation to your data?
The R squared coefficient is 0.86, which means that the model does a good job of explaining the variance in the data.
2. Is your model accurate? Why or why not?
This model is accurate, both because of its high R squared value, but also because of its high r value of ~0.92, which indicates
a very strong negative correlation between the independent and dependent variables.
3. What does the model predict a 10-year-old car with 89000 miles is worth? What about a car that is 20 years old with 150000 miles?
The model predicts that the first car is worth $9257, and the second is worth $2606
4. You may notice that some of your predicted results are negative. This is occurring when the value of age and the mileage of the car are very high. Why do you think this is happening?

This may be happening because of the unbounded linear nature of the model, where, in this case, once the x-axis values go high enough, it is inevitable that the y will dip below the x-axis. Since there is no actual data for these values to reference, the linear nature
of the model necessitates that negative value.