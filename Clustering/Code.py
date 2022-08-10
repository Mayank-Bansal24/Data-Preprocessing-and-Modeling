# =============================================================================
# Name_Mayank Bansal
# Roll Number_B20156
# Mobile Number_9636993445
# Email Id_b20156@students.iitmandi.ac.in
# =============================================================================



from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")

def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(contingency_matrix)
        # Find optimal one-to-one mapping between cluster labels and true labels
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        # Return cluster accuracy
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

#Q1
# Importing data
data = pd.read_csv("Iris.csv")
# Dropping Class Attribute:
data_redu = data.drop(['Species'], axis=1)
standard = preprocessing.scale(data_redu)
# Covariance Matrix:
t = pd.DataFrame(standard).cov()
eigen_values, eigen_vectors = np.linalg.eig(t)
plt.bar(['1', '2', '3', '4'], eigen_values, width=0.5, color="#00CDCD")
plt.xlabel("Components")
plt.ylabel("EigenValue")
plt.title("Eigenvalue vs. Components")
# Below Function show exact number above each bar of graph:
for i in range(len(eigen_values)):
    plt.text(x=i - 0.1, y=(eigen_values[i] + 0.001), s="%.4f" % eigen_values[i], size=9)
plt.show()

#Q2
#read data
data2=pd.read_csv('Iris.csv')   
#list of species     
list1=list(data2['Species'])  
data2=data2.drop('Species',axis=1)    
list2=[]                   
#sort list and list numbers based on species
for i in range(len(list1)):
    if (list1[i]=='Iris-setosa'):
        list2.append(0)
    if (list1[i]=='Iris-versicolor'):
        list2.append(1)
    if (list1[i]=='Iris-virginica'):
        list2.append(2)
#build model
pca=PCA(n_components=2)
#fit data on components
pca.fit(data2)
#coordinates for data2
reduced_data=pca.fit_transform(data2)      
df2=pd.DataFrame(reduced_data)
df3=df2.copy()
#kmeans model
kmeans=KMeans(n_clusters=3)    
predicted=kmeans.fit_predict(df3)
#new column based on cluster no  
df3['Cluster']=predicted
#scatter clusters
plt.scatter(df3.loc[df3['Cluster']==0,0],df3.loc[df3['Cluster']==0,1])
plt.scatter(df3.loc[df3['Cluster']==1,0],df3.loc[df3['Cluster']==1,1])
plt.scatter(df3.loc[df3['Cluster']==2,0],df3.loc[df3['Cluster']==2,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='+',color='black')
plt.show()
#print result
print('Distortion Measure:',kmeans.inertia_)
print('Purity Score:',purity_score(list2,predicted))


#Q3      
#read data
df=pd.read_csv("Iris.csv")      
#dataframe without the species column                                        
d0=df.iloc[:,:4]                                           
                        
#reduced dimension=2                    
pca=PCA(n_components=2)                                  
d0_reduced=pca.fit_transform(d0)                                                                                                            
# the eigen value and vector            
ev,evec=np.linalg.eig(d0.corr())                                                                            
# will contain principal component. i.e,the data resolved along the principal component                                                                                                                                                                                                                      #
p0=[]
# principal component-2            
p1=[]                                                                    
for i in d0_reduced:                                                                                        
    p0.append(i[0])                                                                                         
    p1.append(i[1])                                                                                       
  

# y_actual initially contains the species' proper name
y_actual=df['Species']
for i in range(150):
    if y_actual[i]=="Iris-versicolor":
        y_actual[i]=0
    if y_actual[i]=="Iris-setosa":
        y_actual[i]=1
    if y_actual[i]=="Iris-virginica":
        y_actual[i]=2
# species codefied in 0,1,2


# Part-i
#distortion and purity list
distortion_list=[]                 
purity_list=[]
for i in [2,3,4,5,6,7]:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(d0_reduced)
    kmeans_prediction = kmeans.predict(d0_reduced)
    D={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[]}
    for x in range(len(d0_reduced)):
        D[str(kmeans_prediction[i])].append(d0_reduced[x])
    # kmeans_inertia returns the distortion measure
    distortion_list.append(kmeans.inertia_)
    purity=purity_score(y_actual,kmeans_prediction)
    purity_list.append(purity)
#plot distortion list
plt.plot([2,3,4,5,6,7],distortion_list)
plt.xlabel(" Number of clusters (k)")
plt.ylabel(" Distortion values")
plt.title("Distortion values vs K")
plt.show()
#print purity scores
print("purity scores:",purity_list)

#Q4
data4=pd.DataFrame(reduced_data)
K = 3
gmm = GaussianMixture(n_components = K)
gmm.fit(reduced_data)
gmmpredict=gmm.predict(reduced_data)
data4['Cluster']=gmmpredict
plt.figure(dpi=1200)
plt.scatter(data4.loc[data4['Cluster']==0,0],data4.loc[data4['Cluster']==0,1],marker='*')
plt.scatter(data4.loc[data4['Cluster']==1,0],data4.loc[data4['Cluster']==1,1],marker='x')
plt.scatter(data4.loc[data4['Cluster']==2,0],data4.loc[data4['Cluster']==2,1])
plt.scatter(gmm.means_[:,0],gmm.means_[:,1],marker='+',color='black')
plt.legend()
plt.show()

# Part(b)
print("Distortion measure:",gmm.score(reduced_data))

# Part(c)
print("Purity score:",purity_score(df['Species'],data4['Cluster']))

#Q5
pca = PCA(n_components=2)
pca.fit(standard)
df_pca = pca.fit_transform(standard)
# dis_error and purity stores distortion measure and purity score values respectively for different values of K:
dis_error = []
purity = []
# K value:
k_values = [i for i in range(1, 8)]
for k in k_values:
   # Initialising, fitting and predicting data points using GMM:
   g = GaussianMixture(n_components=k)
   g.fit(df_pca)
   GMM_prediction = g.predict(df_pca)
   dis_error.append(np.sum(g.score_samples(df_pca)))
   purity.append(purity_score(data['Species'], GMM_prediction))
# Plotting Distortion error Vs K Value:
plt.plot(k_values, dis_error, color="#EE3B3B")
plt.scatter(k_values, dis_error)
for i in range(len(k_values)):
    plt.text(x=k_values[i], y=dis_error[i], s="%.3f" % (dis_error[i]))
plt.xlabel("Number of Cluster K")
plt.ylabel('Total data log likelihood')
plt.title("K Value vs total data log likelihood", loc='left')
plt.show()

print("Purity score for different values of K")
for i in range(len(purity)):
   print("For K = %i" % k_values[i], "purity score = %.3f" % (purity[i]))

#Q6
df=pd.read_csv("Iris.csv")
df1=df.iloc[:,0:4]
df1=(df1-df1.mean())/df1.std()
df1C=df1.cov()
df2=df1
eval,evec=np.linalg.eigh(df1C)
evalsl=eval.tolist()
print("Eigenvalues are {},{} ".format(eval[2],eval[3]))
pca=PCA(n_components=2)
pca.fit(df2)
df4=pca.transform(df1)
print(pca.explained_variance_)
red_data=pd.DataFrame(df4,columns=['1','2'])
e_p_s=[1,5]
min_p=[4,10]
for i in e_p_s:
    for j in min_p:
        dbscan_model=DBSCAN(eps=i, min_samples=j).fit(red_data)
        DBSCAN_predictions = dbscan_model.labels_
        core_samples_mask = np.zeros_like(dbscan_model.labels_, dtype=bool)
        core_samples_mask[dbscan_model.core_sample_indices_] = True
        n_clusters_ = len(set(DBSCAN_predictions))- (1 if -1 in DBSCAN_predictions else 0)
        n_noise_ = list(DBSCAN_predictions).count(-1)
        print(n_clusters_)
        unique_labels = set(DBSCAN_predictions)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:                 # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = DBSCAN_predictions == k
            xy = red_data[class_member_mask & core_samples_mask]
            plt.plot(xy.iloc[:, 0],xy.iloc[:, 1],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=14,)

            xy = red_data[class_member_mask & ~core_samples_mask]
            plt.plot(xy.iloc[:, 0],xy.iloc[:, 1],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=6,)

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()
        plt.figure()
