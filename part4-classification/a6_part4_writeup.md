# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
The model is not particuarly accurate, with an r value of 0.65. This is probably because the three independent variables are
on completely different scales from each other, throwing off the model as it cannot properly associate them all with the dependent variable at the same time.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
The model is very accurate, with an r value of 0.89. This is accurate enough for the given use case of predicting purchases, given
that it's just generally accurate.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
The model generally did well, and I can't identify a pattern to the inputs the model was incorrect about.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
Yes.

