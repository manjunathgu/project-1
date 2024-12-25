# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import tree
# Loading the dataset
df = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=35)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"DECISION TREE : Accuracy on Test Data: {accuracy*100}%")

clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
clf3 = clf3.fit(X,y)
predictions = clf3.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"RANDOM FOREST: Accuracy on Test Data: {accuracy*100}%")
pickle.dump(clf3, open('dt.pkl', 'wb'))
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=classifier.classes_)
disp.plot()
plt.show()
