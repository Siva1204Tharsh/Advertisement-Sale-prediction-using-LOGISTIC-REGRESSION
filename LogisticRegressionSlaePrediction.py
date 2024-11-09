# Importing the libraries
import pandas as pd
import numpy as np

# Load the dataset
dataset=pd.read_csv("DigitalAd_dataset.csv")

# # Print the dataset details
# print(dataset.head())
# print(dataset.shape)
# print(dataset.tail())

#Segregate dataset into X and Y
X=dataset.iloc[:,:-1] # X is the feature set
Y=dataset.iloc[:,-1] # Y is the target variable

#Split the dataset into training and testing set
from sklearn.model_selection import train_test_split # Importing the train_test_split function
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

#Feature Scaling
# Importing the StandardScaler function
# we scale our data to make all the features contribute equally to the results
# Fit_Transform -fit method  is calculating the mean and std of the data
#Transform - Transform method is transforming all the features using the respective mean and std
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Logistic Regression and model training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

#Predicting the Sales
Y_pred = model.predict(X_test) # orginal output - Y_test
# print(np.concatenate((Y_pred.values.reshape(len(Y_pred),1), Y_test.values.reshape(len(Y_test),1)), axis=1))
print(Y_pred)

# evaluate the model
from sklearn.metrics import accuracy_score
print("Accuracy Score is : ", accuracy_score(Y_test, Y_pred)*100)

#Predicting the Sales
age=int(input("Enter the age of the customer : "))
salary=int(input("Enter the salary of the customer : "))
newCustomer=[[age,salary]]

result=model.predict(sc_X.transform(newCustomer)) # predict function returns values 
print("Predicted Sales is : ", result)
if result==1:
    print("Customer wiil buy the product")
else:    
    print("Customer will not buy the product")


## Predicting the Sales