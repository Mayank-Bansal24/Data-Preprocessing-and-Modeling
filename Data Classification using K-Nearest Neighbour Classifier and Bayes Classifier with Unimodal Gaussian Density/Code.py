"""
Created on Sat Oct 3 11:08:38 2021
@author: Mayank
Rollno:B20156
PhoneNo: +919636993445
"""
#import needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
import statistics as st
#fiter warning
import warnings
warnings.filterwarnings("ignore")



#Q1
print("Q1")
#importcsv file
df = pd.read_csv('C:/Users/lenovo/Desktop/C++/Lab5/SteelPlateFaults-2class.csv')
#column list
col_list=list(df.columns)
df_class=df['Class']
#copy dataframe
df1=df.copy(deep=True)
#split test and train data
[df1_train, df1_test, df1_class_train, df1_class_test] = train_test_split(df1,df_class, test_size=0.3, random_state=42, shuffle=True)
#dataframe to csv file (train and test)
df1_train.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/SteelPlateFaults-train.csv')
df1_test.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/SteelPlateFaults-test.csv')
#Accuracy list
Accuracy1=[]
for i in (1,3,5):
    #define classifier
    classifier = KNeighborsClassifier(n_neighbors = i)
    #plotting train data
    classifier.fit(df1_train,df1_class_train)
    #predicting test data based on i nearest neighbours
    y_pred = classifier.predict(df1_test)
    #plot confusion matrix
    conf_matrix = confusion_matrix(df1_class_test, y_pred)
    #print result
    print("Neighbours consisted in KNN",i)
    print(conf_matrix)
    print ("Accuracy : ", accuracy_score(df1_class_test, y_pred))
    #append the accuracy obtained
    Accuracy1.append(accuracy_score(df1_class_test, y_pred))


#Q2
print("Q2")
#copy of dataframe
df2=df.copy(deep=True)
#Normalisation for all columns except class
for col in col_list:
    if col!='class':
        for x in range(0,len(df2[col])):
            mx=df2[col].max()
            mn=df2[col].min()
            df2[col][x]= (df2[col][x]-mn)/(mx-mn)
#split test and train data
[df2_train, df2_test, df2_class_train, df2_class_test] = train_test_split(df2,df_class, test_size=0.3, random_state=42, shuffle=True)
#dataframe to csv file (train and test)
df2_train.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/SteelPlateFaults-train-Normalized.csv')
df2_test.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/SteelPlateFaults-test-Normalized.csv')
#Accuracy list
Accuracy2=[]
for i in (1,3,5):
    #define classifier
    classifier = KNeighborsClassifier(n_neighbors = i)
    #plotting train data
    classifier.fit(df2_train,df2_class_train)
    #predicting test data based on i nearest neighbours
    y_pred = classifier.predict(df2_test)
    #plot confusion matrix
    conf_matrix = confusion_matrix(df2_class_test, y_pred)
    #print result
    print("Neighbours consisted in KNN",i)
    print(conf_matrix)
    print ("Accuracy : ", accuracy_score(df2_class_test, y_pred))
    #append the accuracy obtained
    Accuracy2.append(accuracy_score(df2_class_test, y_pred))


#Q3
#call data
train = pd.read_csv("SteelPlateFaults-train.csv")
test = pd.read_csv("SteelPlateFaults-test.csv")
#create a sep label for actual value of class
X_label_test=test['Class']
#define test and train
train = train[train.columns[1:]]
test = test[test.columns[1:]]
test = test[test.columns[:-1]]
#drop bad columns
train.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400'], inplace=True)
test.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400'], inplace=True)
#create separate dataframe for each class
train0 = train[train["Class"] == 0]
train1 = train[train["Class"] == 1]
x_train0 = train0[train0.columns[:-1]]
x_train1 = train1[train1.columns[:-1]]
#create covariance and mean for each data
#convert dataframe further into csv file
cov0 = np.cov(x_train0.T)
arr = pd.DataFrame(cov0)
arr.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/Cov_0.csv')
cov1 = np.cov(x_train1.T)
arr = pd.DataFrame(cov1)
arr.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/Cov_1.csv')
mean0 = np.mean(x_train0)
arr = pd.DataFrame(mean0)
arr.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/mean_0.csv')
mean1 = np.mean(x_train1)
arr = pd.DataFrame(mean1)
arr.to_csv('C:/Users/lenovo/Desktop/C++/Lab5/mean_1.csv')
#define likelihood based on test tuple, Covariance matrix and mean vector
def likelihood(xv, mv, cmat):
    mat = np.dot((xv-mv).T, np.linalg.inv(cmat))
    ins = -0.5*np.dot(mat, (xv-mv))
    ex = np.exp(ins)
    return (ex/((2*np.pi)*12.5 * (abs(np.linalg.det(cmat)))*.5))

#define prior probability
prior0 = len(train0)/len(train)
prior1 = len(train1)/len(train)
#define a list for predicted value of class
pre = []
for i, row in test.iterrows():
    p0 = likelihood(row, mean0, cov0)*prior0
    p1 = likelihood(row, mean1, cov1)*prior1
    #test probability of which class is big and append the value in list
    if(p0 > p1):
        pre.append(0)
    else:
        pre.append(1)
#print result
print()
print("Q3")
print()
bayes_acc = accuracy_score(X_label_test, pre)
print("confusion matrix for bayes classifier : \n",
      confusion_matrix(X_label_test, pre))
print("Accuracy percentage for bayes classifier :",
      bayes_acc)
print()


#Q4
print("Q4")
#print max accuracy obtained in each the 3 ques above
print("Max accuracy for KNN",np.amax(Accuracy1))
print("Max accuracy for KNN Normalized",np.amax(Accuracy2))
print("Accuracy of Bayes Classifier",bayes_acc)
