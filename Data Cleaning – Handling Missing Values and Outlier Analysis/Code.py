# -*- coding: utf-8 -*-
"""Created on Sat Sep 04 12:19:16 2021
@author: Mayank Bansal
Roll no: B20156
Mob.no:  +919636993445
"""

#importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st


#importing data from  both csv files in form of datasheet
Datam = pd.read_csv('landslide_data3_miss.csv')
Datao = pd.read_csv('landslide_data3_original.csv')
#list created with titles of csv as input
l=["dates","stationid","temperature","humidity","pressure","rain","lightavgw/o0","lightmax","moisture"]


#1
A1=[]
for x in l:
#function for finding sum of missing cells  in an attributr
    A1.append(Datam[x].isnull().sum())
#Plotting list of heading and sum of missing cells
plt.bar(l,A1)
plt.xticks(rotation=45)
#show plot
plt.show()


#2a
#sum of tuples where stationid is missing
k=Datam["stationid"].isnull().sum()
print("No of tuples where StationID is missing:",k)
#drop those tuples
Datam=Datam.dropna(subset=["stationid"])

#2b
print("Length of columns in Datafram:",len(Datam["dates"]))
#Dropping tuples where 6 or less no of attributes are written
Datam =Datam.dropna(axis=0, thresh=6)
#lenght of dataframe after dropping tuples
print("Length of columns in Datafram after dropping tuples with 3 or more attributes missing:",len(Datam["dates"]))


#3
c=0
#accessing list of headings
for x in l:
#sum of missing cells for each heading
    print("No of values missing in",x,":",Datam[x].isnull().sum())
#summing each sum for particular attribute
    c+=Datam[x].isnull().sum()
print("Total number of values missing in Dataframe after droppiing tuples:",c)


#4a
#Create a copy of datframe with missing values
Df1=Datam.copy(deep=True)
#Function of mean with condition that cell has a value
def meanm(lista):
    c=0
    len=0
    for k in lista:
#Condition that cell is filled
        if pd.isnull(k)==False:
            c+=float(k)
            len+=1
    return c/len
#fill missing cells with mean of attribute
for i in range(2,9):
    Df1[l[i]] = Df1[l[i]].fillna(meanm(Df1[l[i]]))
    
#i)
#accessing list of heading and printing mean, mode, median and stdev using statistics module
for i in range(2,9):
    print("Mean of",l[i],"after mean substituttion:",st.mean(Df1[l[i]]))
    print("Mean of",l[i],"Original list",st.mean(Datao[l[i]]))  
    print("Median of",l[i],"after mean substitution:",st.median(Df1[l[i]]))
    print("Median of",l[i],"Original list:",st.median(Datao[l[i]]))    
    print("Mode of",l[i],"after mean substitution:",st.multimode(Df1[l[i]]))
    print("Mode of",l[i],"Original list",st.multimode(Datao[l[i]]))
    print("Standard deviation of",l[i],"after mean substitution:",st.stdev(Df1[l[i]]))
    print("Standard deviation of",l[i],"Original list",st.stdev(Datao[l[i]]))
    print("")
    print("")
#ii)
A4=[]
A4b=[]
for i in range(2,9):
    k=0
    c=0
    for x in range(0,len(Datam[l[i]])):
#Condition cell is empty
        if pd.isnull(Datam.iloc[x][l[i]])==True:
            c+=1
#Implementing RMSE for each resp. element
            k+=((Df1.iloc[x][l[i]]-Datao.iloc[x][l[i]])**2)
            #appending RMSE value of each attribute 
            A4.append((k/c)**0.5)
#appending titles needed
            A4b.append(l[i])
    print("RMSE Value of",l[i],"is:",(k/c)**0.5)
#Plotting RMSE and titles
plt.bar(A4b,A4)
plt.xticks(rotation=45)
#Give title
plt.title("RMSE plot after Mean substitution")
#show plot
plt.show()
print("")
print("")    

#4b
#Create copy of dataframe with missing values
Df2=Datam.copy(deep=True)
#Interpolating the dataframe
Df2=Df2.interpolate()
#i)
#accessing list of heading and printing mean, mode, median and stdev using statistics module
for i in range(2,9):
    print("Mean of",l[i],"after interpolation:",st.mean(Df2[l[i]]))
    print("Mean of",l[i],"Original list",st.mean(Datao[l[i]]))  
    print("Median of",l[i],"after interpolation:",st.median(Df2[l[i]]))
    print("Median of",l[i],"Original list:",st.median(Datao[l[i]]))    
    print("Mode of",l[i],"after interpolation:",st.multimode(Df2[l[i]]))
    print("Mode of",l[i],"Original list",st.multimode(Datao[l[i]]))
    print("Standard deviation of",l[i],"after interpolation:",st.stdev(Df2[l[i]]))
    print("Standard deviation of",l[i],"Original list",st.stdev(Datao[l[i]]))
    print("")
    print("")
#ii)
B4=[]
B4b=[]
for i in range(2,9):   
    k=0
    c=0
    for x in range(0,len(Datam[l[i]])):
#Condition cell is empty
        if pd.isnull(Datam.iloc[x][l[i]])==True:
            c+=1
#Implementing RMSE for each resp. element
            k+=((Df2.iloc[x][l[i]]-Datao.iloc[x][l[i]])**2)
    #appending RMSE value of each attribute 
            B4.append((k/c)**0.5)
#appending titles needed
            B4b.append(l[i])
    print("RMSE Value of",l[i],"is:",(k/c)**0.5)
#Plotting RMSE and titles
plt.bar(B4b,B4)
plt.xticks(rotation=45)
#Give title
plt.title("RMSE plot after interpolation")
#show plot
plt.show()
print("")
print("")


#5a
#plot boxplot of temperature
plt.boxplot(Df2["temperature"])
#show grid
plt.grid()
#give title
plt.title("Boxplot of temperature")
plt.plot()
#show plot
plt.show()
#Find 1st IQ
t25=Df2["temperature"].describe()["25%"]
#find 3rd IQ
t75=Df2["temperature"].describe()["75%"]
#IQR
Iqrt=t75-t25
#Find Upper and Lower bound
upt=(Iqrt*1.5)+t75
lwt=t25-(Iqrt*1.5)
T=[]
for x in Df2["temperature"]:
#Implementing function for value out of IQR
    if x>=upt:
        T.append(x)
    elif x<=lwt:
        T.append(x)
#No of outliers
print("Number of outliers in temperature",len(T))
#print outliers
print("Outliers",T)

#plot boxplot of rain
plt.boxplot(Df2["rain"])
#show grid
plt.grid()
#give title
plt.title("Boxplot of rain")
plt.plot()
#show plot
plt.show()
#Find 1st IQ
r25=Df2["rain"].describe()["25%"]
#find 3rd IQ
r75=Df2["rain"].describe()["75%"]
#IQR
Iqrr=r75-r25
#Find Upper and Lower bound
upr=(Iqrr*1.5)+r75
lwr=r25-(Iqrr*1.5)
R=[]
for i in Df2["rain"]:
#Implementing function for value out of IQR
    if i>=upr:
        R.append(i)
    elif i<=lwr:
        R.append(i)
#No of outliers
print("Number of outliers in rain",len(R))  
#print outliers
print("Outliers",R)      

#5b
#Dataframe to list
m=Df2["temperature"].tolist()
#Value of outliers for 5a
for k in T: 
    for x in range(0,len(m)):
#Check if item is outlier
        if m[x]==k:
#Exchanging element with median    
            m[x]=st.median(Df2["temperature"])
#Plot list with removed outliers
plt.boxplot(m)
plt.grid()
#give title
plt.title("Boxplot of temperature")
plt.plot()
#Show plot
plt.show()
#List to Dataframe
m=pd.DataFrame(m,columns=['m1'])
#1st IQ
t251=m['m1'].describe()["25%"]
#3rd IQ
t751=m['m1'].describe()["75%"]
#IQR
Iqrt1=t751-t251
#Upper and lower bound
upt1=(Iqrt1*1.5)+t751
lwt1=t251-(Iqrt1*1.5)
T1=[]
for x in m['m1']:
#implement conditions
    if x>=upt1:
        T1.append(x)
    elif x<=lwt1:
        T1.append(x)
#No of outliers
print("Number of outliers in temperature after deleting outliers",len(T1))
#print outliers
print("Outliers",T1)

#Dataframe to list
n=Df2["rain"].tolist()
#Value of outliers for 5a
for k in R: 
    for x in range(0,len(n)):
#Check if item is outlier
        if n[x]==k:
#Exchanging element with median  
             n[x]=st.median(Df2["rain"])
#Plot list with removed outliers
plt.boxplot(n)
plt.grid()
#give title
plt.title("Boxplot of rain")
plt.plot()
#Show plot
plt.show()
#List to Dataframe
n=pd.DataFrame(n,columns=['n1'])
#1st IQ
r251=n['n1'].describe()["25%"]
#3rd IQ
r751=n['n1'].describe()["75%"]
#IQR
Iqrr1=r751-r251
#Upper and lower bound
upr1=(Iqrr1*1.5)+r751
lwr1=r251-(Iqrr1*1.5)
R1=[]
for x in n['n1']:
#implement conditions
    if x>=upr1:
        R1.append(x)
    elif x<=lwr1:
        R1.append(x)
#No of outliers
print("Number of outliers in rain after deleting outliers",len(R1))
#print outliers
print("Outliers",R1)        
