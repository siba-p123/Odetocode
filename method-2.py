#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import arange
import pandas as pd
import io
import csv
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score


# In[2]:


df = pd.read_csv('dataset_train.csv', names =['smiles', 'propA','propB','propC','propD'])
df.head()


# In[3]:


#Convert the smiles to canonical_smiles inorder to make unbiased smiles
def canonical_smile(smiles):
    mols=[Chem.MolFromSmiles(smi) for smi in smiles]
    smiles=[Chem.MolToSmiles(mol) for mol in mols]
    return smiles


# In[4]:


Canon_SMILES = canonical_smile(df.smiles)
len(Canon_SMILES)


# In[5]:


#replace the smile with canonical_smiles
df['smiles']=Canon_SMILES
df


# In[6]:


#check for duplicate smiles in the smiles list
#duplicate = df[df['smiles'].duplicated()]['smiles'].values
#duplicate


# In[7]:


def morgan_fpt(data):
    Morgan_fpt=[]
    for i in data:
        mol=Chem.MolFromSmiles(i)
        fpts=AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts=np.array(fpts)
        Morgan_fpt.append(mfpts)
    return np.array(Morgan_fpt) 


# In[8]:


Morgan_fpt=morgan_fpt(df['smiles'])
Morgan_fpt.shape


# In[9]:


finger_pt=pd.DataFrame(Morgan_fpt,columns=['col{}'.format(i) for i in range(Morgan_fpt.shape[1])])
finger_pt.shape


# In[10]:


y1 = df['propA'].values
y2 = df['propB'].values
y3 = df['propC'].values
y4 = df['propD'].values
X_train1, X_test1, y_train1, y_test1 = train_test_split(finger_pt, y1, test_size=.2, random_state=2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(finger_pt, y2, test_size=.2, random_state=2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(finger_pt, y3, test_size=.2, random_state=2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(finger_pt, y4, test_size=.2, random_state=2)


# In[11]:


def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    
    plt.figure(figsize=(15, 10))
    plt.plot(prediction[:300], "blue", label="prediction", linewidth=1.0)
    plt.plot(y_test[:300], 'green', label="actual", linewidth=1.0)
    plt.legend()
    plt.ylabel('propA')
    plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
    plt.show()
    
    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse,4))


# In[12]:


#For propertA
model1 = RandomForestRegressor(n_estimators=100)
model1.fit(X_train1, y_train1)
y_pred1 = model1.predict(X_test1)
#Evaluate results
evaluation(model1, X_test1, y_test1)
#For propertB
model2 = RandomForestRegressor(n_estimators=100)
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
#Evaluate results
evaluation(model2, X_test2, y_test2)
#For propertC
model3 = RandomForestRegressor(n_estimators=100)
model3.fit(X_train3, y_train3)
y_pred3 = model3.predict(X_test3)
#Evaluate results
evaluation(model3, X_test3, y_test3)
#For propertD
model4 = RandomForestRegressor(n_estimators=100)
model4.fit(X_train4, y_train4)
y_pred4 = model4.predict(X_test4)
#Evaluate results
#evaluation(model4, X_test4, y_test4)
#plt.scatter(X_train4[:,1], y_train4, color = 'red')
#plt.plot(X_test4[:,1], y_pred4, color = 'green')
#plt.title('Random Forest Regression')
#plt.xlabel('RM')
#plt.ylabel('Price')
#plt.show()


# In[13]:


df2 = pd.read_csv('dataset_test.smi', names=['smile'])
print(df2.head())


# In[14]:


Morgan_fpt_pred=morgan_fpt(df2['smile'])
Morgan_fpt_pred.shape
finger_ptp=pd.DataFrame(Morgan_fpt_pred,columns=['col{}'.format(i) for i in range(Morgan_fpt_pred.shape[1])])
finger_ptp.shape


# In[15]:


pred1=model1.predict(finger_ptp)
pred2=model2.predict(finger_ptp)
pred3=model3.predict(finger_ptp)
pred4=model4.predict(finger_ptp)
#plt.scatter(X_train1, y_train1, label='training data')


# In[16]:


df2['propertyA']=pred1
df2['propertyB']=pred2
df2['propertyC']=pred3
df2['propertyD']=pred4


# In[17]:


df3=pd.DataFrame()
df3['propertyA']=pred1
df3['propertyB']=pred2
df3['propertyC']=pred3
df3['propertyD']=pred4
df3.head()


# In[18]:


df3.to_csv("prediction_Odetocode.csv")


# In[19]:


model4.score(X_test4,pred4)


# In[ ]:




