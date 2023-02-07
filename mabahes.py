import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_excel(r'Results-Azad.xlsx')
#print(dataset)
dataset_striped= dataset.drop('Name',inplace=True,axis=1)
dataset_striped= dataset.drop('Gender',inplace=True,axis=1)
dataset_striped= dataset.drop('City',inplace=True,axis=1)
dataset_striped= dataset.drop('Month',inplace=True,axis=1)
dataset_striped= dataset.drop('Day',inplace=True,axis=1)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
dataset_scaled = normalize(dataset)
dataset_scaled = pd.DataFrame(dataset_scaled,columns=dataset.columns)
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms of Student Simularities")  
dend = shc.dendrogram(shc.linkage(dataset_scaled, method='ward'))
plt.show()