# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
   
## Algorithm
1.Import all the necessary python libraries to perform the given SGDClassifier program.

2.Use the Iris datasets from sklearn.datasets for this program.

3.Take x and y input values from the iris dataset.

4.Use y_pred to store predicted values.

5.Calculate the accuracy score for y_test and y_pred.

6.Create the heatmap with attributes for confusion matrix with matplotlib.pyplot attributes.

7.Show the heapmap for confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Harish R
RegisterNumber: 212224230085
*/
```
```
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head()) 
X = df.drop('target',axis=1) 
y=df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train) 
y_pred=sgd_clf.predict(X_test) 
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}") 
cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

![Screenshot 2025-03-25 034222](https://github.com/user-attachments/assets/19b046e6-5bde-4adf-a902-a1e2b6f3e6d2)

![Screenshot 2025-03-25 034236](https://github.com/user-attachments/assets/cc1f395e-00f6-444d-b496-06ef751a5558)

![Screenshot 2025-03-25 034256](https://github.com/user-attachments/assets/465de4eb-d0c5-4d0b-9f76-6a74c0cab476)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
