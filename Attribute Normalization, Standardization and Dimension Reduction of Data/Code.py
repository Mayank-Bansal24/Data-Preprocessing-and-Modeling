# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:08:38 2021
@author: Mayank
Rollno:B20156
PhoneNo: +919636993445
"""



#import needed modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics as st
#fiter warning
import warnings
warnings.filterwarnings("ignore")



#read csv file
df = pd.read_csv('pima-indians-diabetes.csv')
#list of columns
Cols=list(df.columns)



#1
for i in range(0,8):
    #find outliers
    q1=df[Cols[i]].quantile(0.25)
    q3=df[Cols[i]].quantile(0.75)
    #find IQR
    Iqr=q3-q1
    ub=q3+1.5*(Iqr)
    lb=q1-1.5*(Iqr)
    #find median
    k=st.median(df[Cols[i]])
    #replace outliers with median
    for x in range(0,len(Cols[i])):
        if df[Cols[i]][x]>ub:
            df[Cols[i]][x]=k
        if df[Cols[i]][x]<lb:
            df[Cols[i]][x]=k

#1a   
#copy of dataframe     
df1=df.copy(deep=True)
#find min and max
for i in range(0,8):
    mn=df1[Cols[i]].min()
    mx=df1[Cols[i]].max()
    #print Col name
    print(Cols[i])
    #print max and min
    print("Max before Normalisation",mx)
    print("Min before normalisation",mn)
    for x in range(0,len(df1[Cols[i]])):
        #min max normalisation
        df1[Cols[i]][x]= ((((df1[Cols[i]][x]-mn)/(mx-mn))*7)+5)
    #new min and new max
    mn=df1[Cols[i]].min()
    mx=df1[Cols[i]].max()
    #print new min and new max
    print("Max after Normalisation",mx)
    print("Min after normalisation",mn)
    
#1b
#new copy of dataframe
df2=df.copy(deep=True)
#find mean and standard deviation
for i in range(0,8):
    meana=round(df2[Cols[i]].mean(),2)
    sd=round(df2[Cols[i]].std(),2)
    #print Column name
    print(Cols[i])
    #print mean and standard deviation
    print("Mean before Normalisation",meana)
    print("Stdev before normalisation",sd)
    #standardization using mean and stdev
    for x in range(0,len(df2[Cols[i]])):
        df2[Cols[i]][x]= (df2[Cols[i]][x]-meana)/sd
    #ew mean and standard deviation
    meana=round(df2[Cols[i]].mean())
    sd=round(df2[Cols[i]].std())
    #print new mean and standard deviation
    print("Mean after Normalisation",meana)
    print("Stdev  normalisation",sd)



#2a
#data given
mean = [0, 0]
cov = [[13,-3], [-3,5]]
#using data to generate 1000 samples
a2x,a2y = np.random.multivariate_normal(mean, cov, 1000).T
#plot scatter plot
#title
plt.title("Generated Random samples",color='r')
plt.scatter(a2x,a2y)
#show grid
plt.grid(color='grey',linestyle='-.')
#show plot
plt.show()

#2b
#find eigenvalues and eigenvectors
W,V=np.linalg.eig(cov)
print("Eigenvalues are",W)
print("Eigenvectors are",V)
#origin
o=[0, 0]
#Vec1 and Vec2
ev1 = V[:,0]
ev2 = V[:,1]
# This line below plots the 2d points
plt.scatter(a2x,a2y,color=['g'])
#plot eigenvectors
plt.quiver(*o,*ev1, color=['r'], scale=4)
plt.quiver(*o,*ev2, color=['r'], scale=4)
#give title
plt.title("Eigen Vectors")
#show grid
plt.grid(color='grey',linestyle='-.')
#show plot
plt.show()


#2c
#fig1
#projection of points on Vec1
def dot(a,b,vx,vy):
    d= (a*vx + b*vy)/(((vx**2)+(vy**2))**0.5)
    prx=vx*d/(((vx**2)+(vy**2))**0.5)
    pry=vy*d/(((vx**2)+(vy**2))**0.5)
    return prx,pry
#x coordinates and y coordinates for eingenvector 1
V12cx=[]
V12cy=[]
for i in range(0,1000):
    #call function for projection
    prx,pry= dot(a2x[i],a2y[i],ev1[0],ev1[1])
    V12cx.append(prx)
    V12cy.append(pry)
#scatter points of random data
plt.scatter(a2x,a2y,color=['b'])
#plot eigenvectors
plt.quiver(*o,*ev1, color=['r'], scale=10)
plt.quiver(*o,*ev2, color=['r'], scale=4)
#plot projections corresponding to Vector1
plt.scatter(V12cx,V12cy,color=['g'],s=4)
#give title
plt.title("Projection of Vetor1")
#show grid
plt.grid(color='grey',linestyle='-.')
#show plot
plt.show()

#fig2
#x coordinates and y coordinates for eingenvector 2
V22cx=[]
V22cy=[]
for i in range(0,1000):
    #call function for projection
    prx,pry= dot(a2x[i],a2y[i],ev2[0],ev2[1])
    V22cx.append(prx)
    V22cy.append(pry)
#scatter points for random data
plt.scatter(a2x,a2y,color=['b'])
#plot eigenvectors
plt.quiver(*o,*ev1, color=['r'], scale=4)
plt.quiver(*o,*ev2, color=['r'], scale=10)
#plot projections corresponding to Vector 2
plt.scatter(V22cx,V22cy,color=['g'],s=4)
#give title
plt.title("Projection of Vetor2")
#show grid
plt.grid(color='grey',linestyle='-.')
#show plot
plt.show()

#2d
Drecx=[]
Drecy=[]
#get reconstructed data
for i in range(0,1000):
    Drecx.append(V22cx[i]+V12cx[i])
    Drecy.append(V22cy[i]+V12cy[i])
plt.title("Reconstructed data")
#scatter data
plt.scatter(Drecx,Drecy)
plt.show()
ED=[]
for i in range(0,1000):
    #calculate Eucledian distance for each  tuple.
    ED.append(round((((Drecx[i]-a2x[i])**2)+((Drecy[i]-a2y[i])**2))**0.5))
#print list of error
print(ED)



#3a
#drop class
df2=df2.drop(['class'], axis=1)
#correlation matrix of remaining frame
cormat=df2.corr()
cormat=np.array(cormat)
#eigen values and vectors of correlation matrix
W3a,V3a=np.linalg.eig(cormat)
#rearrange vectors in descending order of eigenvalues
for i in range(0,7):
    for x in range(i+1,8):
        if W3a[x]>W3a[i]:
            W3a[x],W3a[i]=W3a[i],W3a[x]
            for k in range(0,8):
                V3a[x][k],V3a[i][k]=V3a[i][k],V3a[x][k]
#print eigenalues and eigenvectors
print(W3a)
print(V3a)
ev1=V3a[:,0]
ev2=V3a[:,1]
#lists for coordinates of projections
aev1=[]
aev2=[]
for i in range(0,len(df2['pregs'])):
    c=0
    k=0
    #calculating coordinates
    for x in range(0,8):
        c+=ev1[x]*df2[Cols[x]][i]
        k+=ev2[x]*df2[Cols[x]][i]
    aev1.append(c)
    aev2.append(k)
#scatter plot
#give title
plt.title("Coordinates with d=8--> l=2")
plt.scatter(aev1,aev2)
plt.grid()
#show plot
plt.show()
#Print eigenvalue1 and eigenvalue2
print("Eigen Value 1",W3a[0])
print("Eigen Value 2",W3a[1])
print("Variance of projection on eigenvector1",st.variance(aev1))
print("Variance of projection on eigenvector2",st.variance(aev2))

#3b
#give title
plt.title("Eigen values in descending vector")
x=np.arange(1,9)
#line plot
plt.plot(x,W3a,'r',marker='o')
#show plot
plt.show()

#3c
error_record=[]
for i in range(1,9):
    pca = PCA(n_components=i, random_state=50)
    pcar = pca.fit_transform(df2)
    pcap=pca.inverse_transform(pcar)
    total_loss=np.linalg.norm((df2-pcap),None)
    error_record.append(total_loss)
    co3 = np.round(np.matmul(np.transpose(pcar),pcar), decimals=3)
    print(i)
    print(co3)

l2 = [i for i in range(1,9)]
plt.title("Reconstruction Error of Pca",size=22)
plt.plot(l2,error_record,'r',marker='o')
plt.xlabel('No of dimenssion (l)')
plt.ylabel('Euclidan distance')
plt.grid(color='grey', linestyle='-.', linewidth=0.7)
plt.show()

#3d
#datframe of restructured values
df4 = np.dot(df2,V3a)
#new cov values
co3 = np.dot(np.transpose(df4),df4)
#original Covariance matrix
print("Original Matrix")
print(cormat)
#new Covariance matrix
print("New Matrix")
print(co3)
