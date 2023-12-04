# Part 2 - Training and Testing Data Writeup

After completing `a6_part2.py` answer the following questions

## Questions to answer

1. What makes this model more effective than the model you created in the previous lesson?
This model is more effective because it uses some of the original data to test the model.

2. What does the R squared coefficient tell you about the model?
The r squared coefficient of ~0.75 tells us that the model does a good job of explaining the variance in the data.

3. Would you say that your model is accurate? What evidence supports your conclusion? Consider the meaning of the predicted and actual values in the context of the chart below from the American Heart Association’s website on understanding blood pressure.
This model is somewhat accurate, with the high r squared in favor of accuracy, but with predictions sometimes off by a blood pressure catagory (e.g. predicted 137, actual 160).