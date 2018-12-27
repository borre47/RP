# coding: utf-8
# ### Minimum Distance Classifier
# In[46]:
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix

# In[47]:
# m can take two values
#m=np.array([40,60]) #range
def extract_by_row(m,A):
    m=np.asarray(m)# numbeMinimum Distance Classifierr
    A=np.asarray(A)  # data set
    shape=A.shape # get the shape of A 150x4
    mode=m.size # find size of m to decide 
    if mode==1:
        Ix=np.random.choice(np.arange(0,shape[0],1),m,replace=False)# Create index selecting "m" random numbers  
    elif mode==2:
        Ix=np.arange(m[0],m[1]) # create index with a specified range
    X=A[Ix] # select elements 
    Iy=np.setdiff1d(np.arange(0,shape[0],1),Ix) # find the complementary index
    Y=A[Iy] # select elements
    return X,Ix,Y,Iy
# In[48]:
def classifier(A,Ia,B,Ib,targets):
    mk = np.empty([3,4])
    train_targets=targets[Ia] # select train targets
    nTargets=np.unique(targets) # number of targets
    for i,elem in enumerate(nTargets): # for each existing labels
        nElements=elem== train_targets  # boolean vector 
        targetsIndex= np.where(nElements!=False)
        mk[i,:]=np.mean(A[targetsIndex],axis=0)
    test_targets=targets[Ib]
    
    
    g=np.empty([3,1])
    predicted_target=np.empty([test_targets.size,1])
    for i,elem in enumerate(test_targets):
        for j,dis in enumerate(mk):
            g[j,:]=np.sum(dis*B[i])- (0.5)*np.sum(dis*dis)
        predicted_target[i,:]=nTargets[np.argmax(g)]
        
    return confusion_matrix(test_targets,  predicted_target)
        
        
       
# In[49]:
iris = datasets.load_iris()
A=iris.data # data set
targets =np.asarray(iris.target) # labels
m=30# number     
X,Ix,Y,Iy=extract_by_row(m,A)
matrisConf=classifier(X,Ix,Y,Iy,targets)











