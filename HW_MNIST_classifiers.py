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

# Your tasks:
# 1. Set up the pipelines for these six classifiers. Use a set of
# reasonable hyperparameters. You don't need to try find the optimal ones.
# Separate features and target
# Use a smaller sample for quicker running time
df_small = dfdigits.sample(n=10000, random_state=42)
X = df_small.iloc[:, 1:].values # Pixel data
y = df_small.iloc[:, 0].values # Labels
# Normalize rows (contrast adjustment)
normalizer = Normalizer()
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Define classifiers with reasonable hyperparameters
classifiers = {
"LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs',
multi_class='multinomial'),
"DecisionTreeClassifier (gini)": DecisionTreeClassifier(criterion='gini',
max_depth=10),
"DecisionTreeClassifier (entropy)": DecisionTreeClassifier(criterion='entropy',
max_depth=10),
"SVC (rbf, gamma='scale')": SVC(gamma='scale'),
"SVC (linear kernel)": SVC(kernel="linear"),
"LinearSVC": LinearSVC(max_iter=1000),
"KNeighborsClassifier (k=5)": KNeighborsClassifier(n_neighbors=5)
}
# Initialize results list
results = []
# 2. Obtain the classification report for each of the six classifiers.
# Train and evaluate each model
for name, model in classifiers.items():
# Start timer
start_time = time.time()
# Create pipeline
pipeline = Pipeline([
('normalize', normalizer),
('classifier', model)
])
# Fit the model
pipeline.fit(X_train, y_train)
# Predict and evaluate
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = report['accuracy']
# Calculate runtime
runtime = time.time() - start_time
# Append results
results.append({
'Model': name,
'Accuracy': accuracy,
'Runtime (s)': runtime,
'Hyperparameters': model.get_params()
})
# 3. Record the runtimes for these models. Tabulate your results. Also include
# in your table the hyperparameters you used in the models.
# Convert results to DataFrame
results_df = pd.DataFrame(results)
# Display results
print(results_df)
for name, model in classifiers.items():
pipeline = Pipeline([
('normalize', normalizer),
('classifier', model)
])
start_time = time.time()
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
runtime = time.time() - start_time
print(f"Model: {name}, CV Mean Accuracy: {cv_scores.mean():.4f}, Runtime:
{runtime:.2f}s")


# %%


