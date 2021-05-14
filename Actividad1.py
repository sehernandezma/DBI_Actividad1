# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:44:20 2021

@author: Sebastian Hernandez
"""
import numpy as np
import matplotlib.pyplot as plt

#Se crean los grupos de puntos con media = mean y cantidadl.<
N1 = 100
mean = [1,1]
cov = [[0.3, 0.2],[0.2, 0.2]]
data1 = np.random.multivariate_normal(mean, cov, N1)
#L = linalg.cholesky(cov)

N2 = 200

mean = [2,1]
cov = [[0.3, 0.2],[0.2, 0.2]]
data2 = np.random.multivariate_normal(mean, cov, N2)
#L = linalg.cholesky(cov)

N3= 150

mean = [3,1]
cov = [[0.3, 0.2],[0.2, 0.2]]
data3 = np.random.multivariate_normal(mean, cov, N3)
#L = linalg.cholesky(cov)



plt.scatter(data1[:,0], data1[:,1], c='black')
plt.scatter(data2[:,0], data2[:,1], c='yellow')
plt.scatter(data3[:,0], data3[:,1], c='green')
plt.show()



X = np.concatenate((data1,data2,data3),axis=0)
from sklearn.cluster import KMeans

nc=5
centroides1=[]
random_state = [0,15,24,30,99]
for i in range(0,nc):
    
    kmeans = KMeans(n_clusters=3, random_state=random_state[i], init = 'random',  max_iter=100).fit(X)
    centroides1.append(kmeans.cluster_centers_)
    print('centroides corrida{}'.format(i+1))
    print(kmeans.cluster_centers_)



x3 = np.random.normal(0, 150, N1+N2+N3) + X[:,0]
res = 0
for i in x3:
    for j in X[:,0]:
        mean1 = x3.mean()
        mean2 = X[:,0].mean()
        res = res + ((mean1)-i) * ((mean2)-j)
        
res = res/x3.size

X1 = np.hstack((X, np.atleast_2d(x3).T))


nc=5
centroides2=[]
# centroides
random_state = [0,15,24,30,99]
for i in range(0,nc):
    kmeans = KMeans(n_clusters=3, random_state=random_state[i], init = 'random',  max_iter=100).fit(X1)
    centroides2.append(kmeans.cluster_centers_)
    print('centroides2 corrida{}'.format(i+1))
    print(kmeans.cluster_centers_)


x4 = np.random.normal(0, 10, N1+N2+N3) + X[:,0]
x6 = np.random.normal(0, 20, N1+N2+N3) + X[:,1]
x5 = np.random.normal(0, 30, N1+N2+N3) + x3

X2 = np.hstack((X, np.atleast_2d(x3).T))
X2 = np.hstack((X2, np.atleast_2d(x4).T))
X2 = np.hstack((X2, np.atleast_2d(x6).T))
X2 = np.hstack((X2, np.atleast_2d(x5).T))

nc=5
centroides3=[]
random_state = [0,15,24,30,99]

for i in range(0,nc):
    #random_state = int(random.random())*100
    kmeans = KMeans(n_clusters=3, random_state=random_state[i], init = 'random',  max_iter=100).fit(X2)
    centroides3.append(kmeans.cluster_centers_)
    print('centroides3 corrida{}'.format(i+1))
    print(kmeans.cluster_centers_)














