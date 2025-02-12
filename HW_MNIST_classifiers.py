#%%[markdown]
#
# # HW - Classifiers on MNIST dataset
# 
# Like last time, we will use the MNIST dataset, but with classifier models 
# this time. I assumed you have the csv file in a "bigdata" folder. 
# We will try all these:
# 
# * LogisticRegression()
# * DecisionTreeClassifier(): either 'gini' or 'entropy', and various max_depth  
# * SVC(): you can try adjusting the gamma level between 'auto', 'scale', 0.1, 5, etc, and see if it makes any difference 
# * SVC(kernel="linear"): having a linear kernel should be the same as the next one, but the different implementation usually gives different results 
# * LinearSVC() 
# * KNeighborsClassifier(): you can try different k values and find a comfortable choice 
# 
# Use Pipeline. I do not feel that we should standardard all the 784 pixels, but 
# let us try normalize the rows. (In this context, it is like adjusting each 
# image to the same contrast.) 
# 
# Notice that if you use all 60k rows of data, some classifiers can take a 
# long time, depending on your hardware. Always use all the cores in your computer. 
# You can use a smaller subset if needed to make the time manageable.
# 
# Your tasks: 
# 
# 1. Set up the pipelines for these six classifiers. Use a set of 
# reasonable hyperparameters. You don't need to try find the optimal ones. 
# 2. Obtain the classification report for each of the six classifiers. 
# 3. Record the runtimes for these models. Tabulate your results. Also include 
# in your table the hyperparameters you used in the models. 
# 
# BONUS challenge:
# If you can combine the pipeline with cross-validation to obtain the runtimes, 
# that would be most desireable. 
# 

#%%
# Sample code to get started: 
# This creates 784 column headers for the df. 
headers = [ 'x'+('00'+str(i))[-3:] for i in range(785)] # 785 data columns
# 'x000 (first column is y target label, value 0-9), x001, x002, etc there are 28x28=784 pixels columns, 

import pandas as pd
import os
filepath = f'.{os.sep}bigdata{os.sep}mnist_train.csv'
dfdigits = pd.read_csv(filepath, names=headers) 

df = pd.read_csv(filepath)


# %%


