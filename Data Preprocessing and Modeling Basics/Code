# -*- coding: utf-8 -*-
"""Created on Fri Aug 20 12:19:16 2021
@author: Mayank Bansal
Roll no: B20156
Mob.no:  +919636993445
"""

#importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

#importing data from csv file in form of datasheet
Data = pd.read_csv('pima-indians-diabetes.csv')

#lists are imported from the datasheet to form lists with their respective names.
pregs=Data['pregs'].tolist()
plas=Data['plas'].tolist()
pres=Data['pres'].tolist()
skin=Data['skin'].tolist()
test=Data['test'].tolist()
bmi=Data['BMI'].tolist()
pedi=Data['pedi'].tolist()
age=Data['Age'].tolist()
clss=Data['class'].tolist()


#All Required fuctions are written below

#defining function for mean
def meana(list):
    s=0
    #loop runs with adding each element divided by total number of elements
    for i in range(0,len(list)):
        s+=list[i]/len(list)
    return round(s,3)


#function for median
def mediana(list):
    list.sort()
    if len(list)%2==1:
        return list[(len(list)-1)/2]
    elif len(list)%2==0:
        return round((list[int((len(list)/2))-1]+list[int(len(list)/2)])/2,3)


#function for mode
def modea(list):
    return st.multimode(list)


#function for Maximum using an earlier pre-defined function.
def maxa(list):
    return max(list)


#function for Minimum using inbuilt function
def mina(list):
    return min(list)


#function for standard deviation
def stdev(list):
    #getting mean of the list by calling the function defined earlier
    mn=meana(list)
    #using loop to add function corresponding to each element to calculate variance  
    sd=0
    for i in range(0,len(list)):
        sd+=((list[i]-mn)**2)/len(list)
    #square root of variance returns standard deviation
    return round(sd**0.5,3)


#function for Correlation of 2 lists lista and listb
def corra(lista,listb):
    #calculating covariance by adding function corresponding to each number
    cov=0
    for i in range(0,len(lista)):
        cov= cov+((lista[i]-meana(lista))*(listb[i]-meana(listb))/len(lista))
    #implementing function Corr=Cov(Stdeva*Stdevb)
    return cov/(stdev(lista)*stdev(listb))

#function of boxplot
def bplt(list):
    #conversion list to array
    y=np.array(list)
    # Creating plot
    plt.boxplot(y)
    #show grid
    plt.grid()
    


#Part1    
#Part1pregs
print("All the data expected for Number of pregnancies")\
#Calling all functions to give data for pregs
print("Mean(pregs):",meana(pregs))
print("Median(pregs):",mediana(pregs))
print("Mode(pregs:", modea(pregs))
print("Minimum(pregs):",mina(pregs))
print("Maximum(pregs):",maxa(pregs))
print("Standard deviation(pregs):",stdev(pregs))
#empty line
print()

#Part1plas
print("All the data expected for Plasma glucose concentration 2 hours in an oral glucose tolerance test")
#Calling all functions to give data for plas
print("Mean(plas):",meana(plas))
print("Median(plas):",mediana(plas))
print("Mode(plas):", modea(plas))
print("Minimum(plas):",mina(plas))
print("Maximum(plas):",maxa(plas))
print("Standard deviation(plas):",stdev(plas))
#empty line
print()

#Part1pres
print("All the data expected for Diastolic blood pressure (mm Hg)")
#Calling all functions to give data for pres
print("Mean(pres):",meana(pres))
print("Median(pres):",mediana(pres))
print("Mode(pres):", modea(pres))
print("Minimum(pres):",mina(pres))
print("Maximum(pres):",maxa(pres))
print("Standard deviation(pres):",stdev(pres))
#empty line
print()

#Part1skin
print("All the data expected for Triceps skin fold thickness (mm)")
#Calling all functions to give data for skin
print("Mean(skin):",meana(skin))
print("Median(skin):",mediana(pres))
print("Mode(skin):", modea(skin))
print("Minimum(skin):",mina(skin))
print("Maximum(skin):",maxa(skin))
print("Standard deviation(skin):",stdev(skin))
#empty line
print()

#Part1test
print("All the data expected for 2-Hour serum insulin (mu U/mL))")
#Calling all functions to give data for test
print("Mean(test):",meana(test))
print("Median(test):",mediana(test))
print("Mode(test):", modea(test))
print("Minimum(test):",mina(test))
print("Maximum(test):",maxa(test))
print("Standard deviation(test):",stdev(test))
#empty line
print()

#Part1bmi
print("All the data expected for Body mass index (weight in kg/(height in m)^2)")
#Calling all functions to give data for BMI
print("Mean(bmi):",meana(bmi))
print("Median(bmi):",mediana(bmi))
print("Mode(bmi):", modea(bmi))
print("Minimum(bmi):",mina(bmi))
print("Maximum(bmi):",maxa(bmi))
print("Standard deviation(bmi):",stdev(bmi))
#empty line
print()

#Part1pedi
print("All the data expected for Diabetes pedigree function")
#Calling all functions to give data for Pedi
print("Mean(pedi):",meana(pedi))
print("Median(pedi):",mediana(pedi))
print("Mode(pedi):", modea(pedi))
print("Minimum(pedi):",mina(pedi))
print("Maximum(pedi):",maxa(pedi))
print("Standard deviation(pedi):",stdev(pedi))
#empty line
print()

#Part1age
print("All the data expected for Age(years)")
#Calling all functions to give data for Age
print("Mean(age):",meana(age))
print("Median(age):",mediana(age))
print("Mode(age):", modea(age))
print("Minimum(age):",mina(age))
print("Maximum(age):",maxa(age))
print("Standard deviation(age):",stdev(age))
#empty line
print()



#Part2a
#Scatter plot of age and pregs
plt.scatter(Data['Age'],Data['pregs'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Pregs")
plt.xlabel('Age(years)-->')
plt.ylabel('Number of times pregnant-->')
#show plot
plt.show()

#Scatter plot of age and plas
plt.scatter(Data['Age'],Data['plas'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Plas")
plt.xlabel('Age(years)-->')
plt.ylabel('Plasma glucose concentration-->')
#show plot
plt.show()

#Scatter plot of age and pres
plt.scatter(Data['Age'],Data['pres'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Pres(mm Hg)")
plt.xlabel('Age(years)-->')
plt.ylabel('Diastolic blood pressure (mm Hg)-->')
plt.show()

#Scatter plot of age and skin
plt.scatter(Data['Age'],Data['skin'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Skin(mm)")
plt.xlabel('Age(years)-->')
plt.ylabel('Triceps skin fold thickness (mm)-->')
#show plot
plt.show()

#Scatter plot of age and test
plt.scatter(Data['Age'],Data['test'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Test(mu U/ml)")
plt.xlabel('Age(years)-->')
plt.ylabel('2-Hour serum insulin(mu U/ml)-->')
plt.show()

#Scatter plot of age and BMI
plt.scatter(Data['Age'],Data['BMI'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. BMI(kg/m^2)")
plt.xlabel('Age(years)-->')
plt.ylabel('BMI(kg/m^2)-->')
#show plot
plt.show()

#Scatter plot of age and pedi
plt.scatter(Data['Age'],Data['pedi'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of Age(years) vs. Pedi")
plt.xlabel('Age(years)-->')
plt.ylabel('Diabetes pedigree function-->')
#show plot
plt.show()


#Part2b
#Scatter plot of BMI and pregs
plt.scatter(Data['BMI'],Data['pregs'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Pregs")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Number of times pregnant-->')
#show plot
plt.show()

#Scatter plot of BMI and plas
plt.scatter(Data['BMI'],Data['plas'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Plas")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Plasma glucose concentration-->')
#show plot
plt.show()

#Scatter plot of BMI and pres
plt.scatter(Data['BMI'],Data['pres'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Pres(mm Hg)")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Diastolic blood pressure (mm Hg)-->')
#show plot
plt.show()

#Scatter plot of BMI and skin
plt.scatter(Data['BMI'],Data['skin'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Skin(mm)")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Triceps skin fold thickness (mm)-->')
#show plot
plt.show()

#Scatter plot of BMI and test
plt.scatter(Data['BMI'],Data['test'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Test(mu U/ml)")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('2-Hour serum insulin(mu U/ml)-->')
#show plot
plt.show()

#Scatter plot of BMI and Age
plt.scatter(Data['BMI'],Data['Age'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Age(years)")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Age(years)-->')
#show plot
plt.show()

#Scatter plot of BMI and pedi
plt.scatter(Data['BMI'],Data['pedi'])
#giving title,xlabel and ylabel to plot
plt.title("Scatter plot of BMI(kg/m^2) vs. Pedi")
plt.xlabel('BMI(kg/m^2)-->')
plt.ylabel('Diabetes pedigree function-->')
#show plot
plt.show()



#Part3a
#printing correlation of Age and all other attributes except class by calling previously defined function
print("Values of correlations with Age and other attributes")
print("Correlation coefficient of Age and Pregs is:",round(corra(Data['Age'],Data['pregs']),3))
print("Correlation coefficient of Age and Plas is:",round(corra(Data['Age'],Data['plas']),3))
print("Correlation coefficient of Age and Pres is:",round(corra(Data['Age'],Data['pres']),3))
print("Correlation coefficient of Age and Skin is:",round(corra(Data['Age'],Data['skin']),3))
print("Correlation coefficient of Age and Test is:",round(corra(Data['Age'],Data['test']),3))
print("Correlation coefficient of Age and BMI is:",round(corra(Data['Age'],Data['BMI']),3))
print("Correlation coefficient of Age and Pedi is:",round(corra(Data['Age'],Data['pedi']),3))
print("Correlation coefficient of Age and Age is:",round(corra(Data['Age'],Data['Age']),3))
print("")


#Part3b
#printing correlation of BMI and all other attributes except class by calling previously defined function
print("Values of correlations with BMI and other attributes")
print("Correlation coefficient of BMI and Pregs is:",round(corra(Data['BMI'],Data['pregs']),3))
print("Correlation coefficient of BMI and Plas is:",round(corra(Data['BMI'],Data['plas']),3))
print("Correlation coefficient of BMI and Pres is:",round(corra(Data['BMI'],Data['pres']),3))
print("Correlation coefficient of BMI and Skin is:",round(corra(Data['BMI'],Data['skin']),3))
print("Correlation coefficient of BMI and Test is:",round(corra(Data['BMI'],Data['test']),3))
print("Correlation coefficient of BMI and Age is:",round(corra(Data['BMI'],Data['Age']),3))
print("Correlation coefficient of BMI and Pedi is:",round(corra(Data['BMI'],Data['pedi']),3))
print("Correlation coefficient of BMI and BMI is:",round(corra(Data['BMI'],Data['BMI']),3))



#Part4a
#list to array
x=np.array(pregs)
#plot graph with 18 bars
plt.hist(x,bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
#giving title, xlabel and ylabel
plt.xlabel("Number of pregnencies-->")
plt.ylabel("Frequency--->")
plt.title("Histogram for number of Prgnencies")
#show grid
plt.grid()
#show plot
plt.show()


#Part4b
#list to array
x=np.array(skin)
#plot graph with 20 bars
plt.hist(x,bins=20)
#giving title ,xlabel and ylabel
plt.xlabel("Triceps skin fold thickness (mm)-->")
plt.ylabel("Frequency--->")
plt.title("Histogram for Triceps skin fold thickness (mm)")
#show grid
plt.grid()
#show plot
plt.show()



#Part5a
#sorting preg based on class=0
pregsc0=[]
for i in range(0,len(pregs)):
    if clss[i]==0:
        pregsc0.append(pregs[i])
#list to array
x0=np.array(pregsc0)
#plot graph with 13 bars
plt.hist(x0,bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
#giving title, xlabel and ylabel
plt.xlabel("Number of pregnencies-->")
plt.ylabel("Frequency--->")
plt.title("Histogram for number of Prgnencies(Class0)")
#show grid
plt.grid()
#show plot
plt.show()


#Part5b
#sort pregs based on class=1
pregsc1=[]
for i in range(0,len(pregs)):
    if clss[i]==1:
        pregsc1.append(pregs[i])
#list to aray
x1=np.array(pregsc1)
#plot graph with 18 bars
plt.hist(x1,bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
#giving title, xlabel and ylabel
plt.xlabel("Number of pregnencies-->")
plt.ylabel("Frequency--->")
plt.title("Histogram for number of Prgnencies(Class1)")
#show grid
plt.grid()
#show plot
plt.show()

#6
#Calling bplt function for boxplot construction
bplt(pregs)
#give title and show plot
plt.title("Number of Pregnancies")
plt.show()

bplt(plas)
#give title and show plot
plt.title("Plasma glucose concentration")
plt.show()

bplt(pres)
#give title and show plot
plt.title("Diastolic blood pressure(mm Hg)")
plt.show()

bplt(skin)
#give title and show plot
plt.title("Triceps skin fold thickness(mm)")
plt.show()

bplt(test)
#give title and show plot
plt.title("2-Hour serum insulin (mu U/mL)")
plt.show()

bplt(bmi)
#give title and show plot
plt.title("Body mass index(kg/m^2)")
plt.show()

bplt(pedi)
#give title and show plot
plt.title("Diabetes pedigree function")
plt.show()

bplt(age)
#give title and show plot
plt.title("Age(years)")
plt.show()


#----------------END---------------
