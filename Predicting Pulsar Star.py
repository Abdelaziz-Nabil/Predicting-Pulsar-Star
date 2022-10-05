#!/usr/bin/env python
# coding: utf-8

# ## About Dataset 
# #### [Problem Statement:](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate)
# > **Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter. Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,which treat the candidate data sets as binary classification problems.**
# 
# #### Attribute Information:
# > Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:
# 
# 1. Mean of the integrated profile.
# 2. Standard deviation of the integrated profile.
# 3. Excess kurtosis of the integrated profile.
# 4. Skewness of the integrated profile.
# 5. Mean of the DM-SNR curve.
# 6. Standard deviation of the DM-SNR curve.
# 7. Excess kurtosis of the DM-SNR curve.
# 8. Skewness of the DM-SNR curve.
# 9. Class
# 
# ### Task
# **Given a Pulsar Star’s information, build a machine learning model that can classify the Star.**

# * [Importing the libraries](#importing-the-libraries)
# * [Reading the data](#Reading-the-data)
# * [Exploring the data](#Exploring-the-data)
# * [Figure out Data](#Figure-out-Data)
# * [Detect Outliers for Every columns](#Detect-Outliers-for-Every-columns)
# * [Scaling and Split the data](#Scaling-and-Split-the-data)
# * [balance the data](#balance-the-data)
# * [Modeling](#Modeling)
# * [compersion between models](#compersion-between-models)

# ## Importing the libraries

# In[321]:


import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn .metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import  mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score
from random import shuffle
import warnings
warnings.filterwarnings('ignore')


# ## Reading the data

# In[239]:


column=['Mean_integrated_profile',
       'std_integrated_profile',
       'Excess_kurtosis_integrated_profile',
       'Skewness_integrated_profile', 'Mean_DM_SNR',
       'std_DM_SNR',
       'Excess_kurtosis_DM_SNR', 'Skewness_DM_SNR',
       'target_class']


# In[240]:


path='D:\Samsung Innovation Campus\Data\pulsar data\HTRU_2.csv'
df=pd.read_csv(path,sep = "," , encoding = "utf-8")


# In[241]:


df.columns=column


# ## Exploring the data

# In[242]:


df.head()


# In[243]:


df.describe()


# **Integrated Profile**
# 
# This curve seems to be a normal distribution. which is symmetric about the mean and regular thickness of tails
# 
# 1. **Mean** : Since mean is close to 50% mark and 25% and 75% are within 1 std. dev. from the mean. Large difference between min and max, hence larger std. dev. Hence the mean of the integrated profile seems volatile with a large standard deviation.
# 
# 2. **Std. Dev** : Since mean is close to 50% mark and 25% and 75% are within less than 1 std. dev. from the mean, this data is normally distributed. The max and min values are 3-4 standard deviations from the mean.
# 
# 3. **Excess Kurtosis** : Majority of this data (atleast 75%) is less than mean. Hence a large head portion in this distribution. Hence the distribution of the left of mean is more tightly spread than the right. This means the integrated profile's tails are generally the same size as normal distributions.
# 
# 4. **Skewness** : Majority of this data (definitely more 75%) is less than mean. Hence a large head portion in this distribution. Hence the distribution of the left of mean is more tightly spread than the right. Hence the integrated profile must not be very skewed.
# 
# **DM-SNR Curve**
# 
# This curve is expected to be more highly spread than a normal distribution, and skewed towards the right of the mean.
# 
# 1. **Mean** : Standard deviation of the mean is very high, with more than 75% values being less than mean. Hence mean of most curves are small
# 
# 2. **Std. Dev** : Very skewed towards the higher side. Can expect most values to have standard deviation lower than 28. However, comparing this with the distribution of DM-SNR curve, which is mostly less than 5. This is a very highly spread DM-SNR curve.
# 
# 3. **Excess Kurtosis** : This is normally distributed since mean = 50% value and every 25% is approx one std. dev. However, these values are high, hence the DM-SNR curve has fatter tails than normal distribution.
# 
# 4. **Skewness** : Very high values, and high standard deviation. Hence the DM-SNR curve is very skewed (expected)

# In[244]:


df.median(axis = 0)


# In[245]:


df.info()


# In[246]:


missing_values_df=df.isna().sum()
missing_values_df


# ### Check if our data is balanced, why not?

# In[247]:


data=df['target_class'].value_counts()
data=round(((data/len(df)) * 100),2)
data=data.to_frame()
data


# In[248]:


fig, axs = plt.subplots(figsize=(9, 6))
fg=sns.countplot(data=df,x='target_class')
fg.set_xlabel("Target Class")
for bars in axs.containers:
        axs.bar_label(bars, fmt='%1.f',padding=5)
fig.savefig('D:\\Samsung Innovation Campus\\Data\\pulsar data\\target_class.png', bbox_inches='tight', dpi=80)
plt.show()


# * **Class imbalance**: (10:1 almost)
# * No. of instances pulsar stars are detected in dataset is 1639
# * No. of instances pulsar stars are not detected in dataset is 16258

# In[249]:


# colors palettes
palett=['Accent','Blues','BrBG','BuGn','BuPu','Dark2','GnBu','Greens','Greys','OrRd','Oranges','PRGn','Paired','Pastel1',
   'Pastel2','PiYG','PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu','RdGy','RdPu','RdYlBu','RdYlGn','Reds','Set1',
   'Set2','Set3','Spectral','YlGn','YlGnBu','YlOrBr','YlOrRd']


# ## Figure out Data

# In[250]:


def create_plot(columne,data,i):
    data=data[df[columne].notnull()]
    fig, ax = plt.subplots(1,2, figsize = (15,5))
    font_dict = {'fontsize': 14}
    title=['Before Distribution','After Distribution']
    ax = np.ravel(ax)
    if i==1:
        sns.set(style='whitegrid')
        sns.distplot(data=data,x=columne ,ax = ax[0],palette='magma').set_title(title[i])
        sns.boxplot(data=data,y=columne ,x='target_class',ax = ax[1],palette='magma').set_title(title[i])
    else:
        sns.set(style='whitegrid')
        sns.kdeplot(data=data,x=columne ,ax = ax[0],color='#2171b5',palette='cubehelix_r').set_title(title[i])
        sns.boxplot(data=data,y=columne ,x='target_class',ax = ax[1],palette='cubehelix_r').set_title(title[i])
        
    ax = np.reshape(ax, (1, 2))
    plt.tight_layout()


# In[251]:


def create_plot(columne,data,i,xlabel):
    data=data[df[columne].notnull()]
    fig, ax = plt.subplots(1,2, figsize = (13,7))
    font_dict = {'fontsize': 14}
    ax = np.ravel(ax)
    global palett
    pale=np.random.choice(palett,1)[0]
    sns.set(style='whitegrid')
    Hist=sns.kdeplot(data=data,x=columne,ax = ax[0],palette=pale,multiple="stack")
    Hist.set_title("Hisotogram of " + xlabel)
    Hist.set_xlabel(xlabel)
    
    box=sns.boxplot(data=data,y=columne ,x='target_class',ax = ax[1],palette=pale)#.set_title("boxplot of " + columne+" with Target")
    box.set_title("boxplot of " + xlabel+" with Target")
    box.set_ylabel(xlabel)
    box.set_xlabel('Target Class')
    
    
    #fig.savefig('D:\\Samsung Innovation Campus\\Data\\pulsar data\\{}.png'.format(xlabel), bbox_inches='tight', dpi=150)
    ax = np.reshape(ax, (1, 2))
    plt.tight_layout()


# In[252]:


def create_scatter_plo(x,y,i,a,b):
    pal=['cubehelix_r','magma']
    data=df[(df[x].notnull()) & (df[y].notnull())]
    data['target_class'] = data.target_class.astype(object)
    color_discrete=["red","green","blue", "goldenrod","magenta"]
    shuffle(color_discrete)
    fig = px.scatter(data, x=x, y =y,color="target_class",symbol="target_class",
                     labels={x:a,y:b,"target_class":"target class"},
                    color_discrete_sequence=color_discrete[:2],
                    )
    #fig.write_image("D:\Samsung Innovation Campus\Data\pulsar data\fig{}.png".format(i))
    fig.show()


# ### Mean of the integrated profile & Mean of the DM-SNR curve

# In[253]:


create_plot(columne='Mean_integrated_profile',data=df,i=0,xlabel='Mean of the integrated profile')


# - left Skewed distribution amost a normal distribution except for a large left tail
# - target 0 has some outliers
# - target 1 Almost no outliers

# In[254]:


create_plot(columne='Mean_DM_SNR',data=df,i=0,xlabel='Mean of the DM-SNR curve')


# - Highly crowded, **Right Skewed** distribution, with a fat right tail.
# - target 0 has outliers
# - target 1 has some outliers

# In[255]:


create_scatter_plo(x='Mean_integrated_profile', y = 'Mean_DM_SNR',i=0,a='Mean of the integrated profile',b='Mean of the DM-SNR curve')


# - From this figure, we can deduce the effect of the "Mean of the integrated profile" in choosing Target Class
# - There is no change in the selection of the Target Class according to the  y-axis Since the single value in most of the figure can be a pulsar stars or not

# ### Standard deviation of the integrated profile & Standard deviation of the DM-SNR curve

# In[256]:


create_plot(columne='std_integrated_profile',data=df,i=1,xlabel='Standard deviation of the integrated profile')


# - Std. Dev Almost a **normal** distribution with a tail to right of mean
# - Std. Dev has some outliers

# In[257]:


create_plot(columne='std_DM_SNR',data=df,i=1,xlabel='Standard deviation of the DM-SNR curve')


# - **Right skewed**, with a fat right tail
# - target 0 has some outliers
# - target 1 Almost no outliers

# In[258]:


create_scatter_plo(x='std_integrated_profile', y = 'std_DM_SNR',i=1,a='Standard deviation of the integrated profile',b='Standard deviation of the DM-SNR curve')


# - The figure shows that each of the properties has a strong specific **range** and is weakly intertwined

# ### Excess kurtosis of the integrated profile & Excess kurtosis of the DM-SNR curve

# In[259]:


create_plot(columne='Excess_kurtosis_integrated_profile',data=df,i=0,xlabel='Excess kurtosis of the integrated profile')


# - **Right skewed**, with few outliers in the form of a long tail to the right
# - **Highly crowded**
# - target **1** Almost **no** outliers

# In[260]:


create_plot(columne='Excess_kurtosis_DM_SNR',data=df,i=0,xlabel='Excess kurtosis of the DM-SNR curve')


#  - Left skewed  Almost **normal** distribution. Minimal tails.
#  - It has some **outliers**

# In[261]:


create_scatter_plo(x='Excess_kurtosis_integrated_profile', y = 'Excess_kurtosis_DM_SNR',i=0,a='Excess kurtosis of the integrated profile',b='Excess kurtosis of the DM-SNR curve')


# - note that most of the small values of the **Excess kurtosis of the integrated profile** on x-axis are not pulsar stars, while the high values are a pulsar stars
# - note that most of the small values of the **Excess kurtosis of the DM-SNR curve** on y-axis are pulsar stars, while the high values are a not pulsar stars

# ### Skewness of the integrated profile & Skewness of the DM-SNR curve

# In[262]:


create_plot(columne='Skewness_integrated_profile',data=df,i=1,xlabel='Skewness of the integrated profile')


# - Highly crowded, few outliers in the form of a long tail to the right
# - It has some outliers, **right skewed**

# In[263]:


create_plot(columne='Skewness_DM_SNR',data=df,i=1,xlabel='Skewness of the DM-SNR curve')


# - Highly right skewed with a long right tail
# - It has some outliers, right skewed 

# In[264]:


create_scatter_plo(x='Skewness_integrated_profile', y = 'Skewness_DM_SNR',i=1,a='Skewness of the integrated profile',b='Skewness of the DM-SNR curve')


# - note that most of the small values of the **Skewness of the integrated profile** on x-axis are not pulsar stars, while the high values are a pulsar stars
# - note that most of the small values of the **Skewness of the DM-SNR curve** on y-axis are pulsar stars, while the high values are a not pulsar stars

# ----

# In[265]:


g = sns.pairplot(df,hue='target_class',corner=True)


# ### Correlation Heatmap

# In[266]:


plt.figure(figsize = (14, 10))
corr_mat = df.corr()
sns.heatmap(corr_mat, xticklabels = corr_mat.columns,cmap='viridis', yticklabels = corr_mat.columns, annot=True)
plt.show()


# ### Correlations between 2 dependent variables
# 
# **Highly positively correlated:**
# 
# 1. Skewness of the integrated profile and Excess kurtosis of the integrated profile
# 2. Skewness of the DM-SNR curve and Excess kurtosis of the DM-SNR curve
# 3. Mean the DM-SNR curve and Standard Deviation of the DM-SNR curve
# 
# **Highly negatively correlated:**
# 
# 1. Mean of the integrated profile and Excess kurtosis of the integrated profile
# 2. Mean of the integrated profile and Skewness of the integrated profile
# 3. Excess kurtosis the DM-SNR curve and Standard Deviation of the DM-SNR curve
# 
# ### Correlations between independent and dependent variable
# **Highly positively correlated:**
# 
# 1. Excess kurtosis of the integrated profile
# 2. Skewness of the integrated profile
# 
# **Highly negatively correlated:**
# 
# 1. Mean of the integrated profil

# ## Detect Outliers for Every columns

# In[267]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# > **Since there are not many outliers we can either remove them or cap them. However, removing them is not advised, so we will cap them using IQR**

# In[268]:


data_df=df.copy()
data0=df[df['target_class']==0.0]
data1=df[df['target_class']==1.0]


# In[269]:


for column in df.iloc[:,:-1].columns:
    lr,ur=remove_outlier(data0[column])
    u_data=(data_df[(data_df['target_class']==0)&(data_df[column] > ur)])[column]
    l_data=(data_df[(data_df['target_class']==0)&(data_df[column] < lr)])[column]
    if not u_data.empty:
        u_data=sorted(u_data)
        index=int(round(len(u_data)*0.7,0))
        ur=u_data[index]
    if not l_data.empty:
        l_data=sorted(l_data)
        index=int(round(len(l_data)*0.3,0))
        lr=l_data[index]
    index=(data_df[(data_df['target_class']==0)&((data_df[column] < lr)|(data_df[column] > ur))]).index
    index=index.to_list()
    data_df.drop(index,inplace=True)
    
    lr,ur=remove_outlier(data1[column])
    u_data=(data_df[(data_df['target_class']==1)&(data_df[column] > ur)])[column]
    l_data=(data_df[(data_df['target_class']==1)&(data_df[column] < lr)])[column]
    if not u_data.empty:
        u_data=sorted(u_data)
        index=int(round(len(u_data)*0.7,0))
        ur=u_data[index]
    if not l_data.empty:
        l_data=sorted(l_data)
        index=int(round(len(l_data)*0.3,0))
        lr=l_data[index]
    index=(data_df[(data_df['target_class']==1)&((data_df[column] < lr)|(data_df[column] > ur))]).index
    index=index.to_list()
    data_df.drop(index,inplace=True)
    


# In[270]:


data_df.shape


# In[271]:


df.shape


# ### Figure out outlier Before remove them and  After

# In[272]:


def box_plot(column,xlabel):
    global palett
    fig, ax = plt.subplots(2,2, figsize = (12,8))
    font_dict = {'fontsize': 10}
    ax = np.ravel(ax)
    shuffle(palett)
    pale=palett[:4]
    
    sns.set(style='whitegrid')
    pl1=sns.boxplot(data=df[df['target_class']==0] ,x=column,ax = ax[0],palette=pale[0])
    pl1.set_xlabel(xlabel+" of 0 target Before")
    
    sns.set(style='whitegrid')
    pl2=sns.boxplot(data=data_df[data_df['target_class']==0] ,x=column,ax = ax[1],palette=pale[1])
    pl2.set_xlabel(xlabel+" of 0 target After")

    
    sns.set(style='whitegrid')
    pl3=sns.boxplot(data=df[df['target_class']==1] ,x=column,ax = ax[2],palette=pale[2])
    pl3.set_xlabel(xlabel+" of 1 target Before")
    
    sns.set(style='whitegrid')
    pl4=sns.boxplot(data=data_df[data_df['target_class']==1] ,x=column,ax = ax[3],palette=pale[3])
    pl4.set_xlabel(xlabel+" of 1 target After")
    
    #fig.savefig('D:\\Samsung Innovation Campus\\Data\\pulsar data\\{}.png'.format(column), bbox_inches='tight', dpi=80)
    ax = np.reshape(ax, (2, 2))
    plt.tight_layout()


# In[273]:


box_plot('Mean_integrated_profile',xlabel='Mean of the integrated profile')


# In[274]:


box_plot('std_integrated_profile',xlabel='Standard deviation of the integrated profile')


# In[275]:


box_plot('Excess_kurtosis_integrated_profile',xlabel='Excess kurtosis of the integrated profile')


# In[276]:


box_plot('Skewness_integrated_profile',xlabel='Skewness of the integrated profile')


# In[277]:


box_plot('Mean_DM_SNR',xlabel='Mean of the DM-SNR curve')


# In[278]:


box_plot('std_DM_SNR',xlabel='Standard deviation of the DM-SNR curve')


# In[279]:


box_plot('Excess_kurtosis_DM_SNR',xlabel='Excess kurtosis of the DM-SNR curve')


# In[280]:


box_plot('Skewness_DM_SNR',xlabel='Skewness of the DM-SNR curve')


# ## Scaling and Split the data

# ### Split

# In[281]:


x = data_df.drop('target_class',axis=1)
y = data_df['target_class'].values


# ### Scaling
# - **StandardScaler** follows Standard Normal Distribution (SND). Therefore, it makes mean = 0 and scales the data to unit variance. 
# - **MinMaxScaler** scales all the data features in the range [0, 1] or else in the range [-1, 1] if there are negative values in the dataset. This scaling compresses all the inliers in the narrow range [0, 0.005].
# 
# In the presence of outliers, StandardScaler does not guarantee balanced feature scales, due to the influence of the outliers while computing the empirical mean and standard deviation. This leads to the shrinkage in the range of the feature values. 
# - By using **RobustScaler()**, we can remove the outliers and then use either StandardScaler or MinMaxScaler for preprocessing the dataset. 

# In[282]:


from sklearn import preprocessing
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df,columns=list(x.columns ))


# In[283]:


scaler = MinMaxScaler()
x = scaler.fit_transform(robust_df)


# ## balance the data

# ### Why we have to balance the data?
# The answer is quite simple, to make our predictions more accurate.  
# 
# Because if we have imbalanced data, the model is more biased to the dominant target class and tends to predict the target as the predominant target class.

# ### Techniques for handling imbalanced data
# For handling imbalance data we are having many other ways, In this article, we will learn about the below techniques along with the code implementation.
# 
# - Oversampling
# - Undersampling

# In[284]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler


# ### Oversampling

# #### RandomOverSampler

# In[285]:


ovs= RandomOverSampler(random_state=42)
x_res , y_res = ovs.fit_resample(x,y)


# In[286]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[287]:


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res, test_size=0.3,random_state = 1234)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train,y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
print("accuracy score",accuracy_score(y_test, y_pred))
print("mean squared error :",men)
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='BuPu')
plt.show()


# #### SMOTETomek

# In[288]:


from imblearn.combine import SMOTETomek
smk = SMOTETomek()
x_res , y_res = smk.fit_resample(x,y)


# In[289]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[290]:


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res, test_size=0.3,random_state = 1234)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train,y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
print("accuracy score",accuracy_score(y_test, y_pred))
print("mean squared error :",men)
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='hot')
plt.show()


# ---

# ### Oversampling pros and cons
# 
# **Pros:**
# - This method doesn’t lead to information loss.
# - Performs  well and gives good accuracy.
# - It creates new synthetic data points with the nearest neighbours from existing data.
# 
# **Cons:**
# - Increase the size of data takes high time for training.
# - It may also lead to overfitting since it is replicating the minor classes.
# - Need extra storage.

# ---

# ### UnderSampling

# In[291]:


from imblearn.under_sampling import RandomUnderSampler


# #### NearMiss  method

# In[292]:


# Under sampling with nearmiss
from imblearn.under_sampling import NearMiss
nm = NearMiss()
x_res,y_res = nm.fit_resample(x,y)


# In[293]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[294]:


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res, test_size=0.3,random_state = 1234)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train,y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
print("accuracy score",accuracy_score(y_test, y_pred))
print("mean squared error :",men)
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='afmhot')
plt.show()


# #### RandomunderSampler method

# In[295]:


# Under sampling with randomundersample
ous = RandomUnderSampler(random_state=42)
x_res,y_res = ous.fit_resample(x,y)


# In[296]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[297]:


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res, test_size=0.3,random_state = 1234)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train,y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
print("accuracy score",accuracy_score(y_test, y_pred))
print("mean squared error :",men)
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='YlOrBr')
plt.show()


# ---

# ### Undersampling pros and cons
# **Pros:**
# - Reduces storage problems, easy to train
# - In most cases it creates a balanced subset that carries the greatest potential for representing the larger group as a whole.
# - It produces a simple random sample which is much less complicated than other techniques.
# 
# **Cons:**
# - It can ignore potentially useful information which could be important for building  classifiers.
# - The sample chosen by random under-sampling may be a biased sample, resulting in inaccurate results with the actual test data.
# - Loss of useful information of the majority class.

# ### When to use oversampling VS undersampling
# We have a fair amount of knowledge on these two data imbalance handling techniques, but we use them as both the methods are for handling the imbalanced data issue.
# 
# - **Oversampling:** We will use oversampling when we are having a limited amount of data.
# - **Undersampling:** We will use undersampling when we have huge data and undersampling the majority call won't effect the data

# ----

# #### so we will use Undersampling : RandomunderSampler method

# ----

# ## Modeling

# In[330]:


compa={"accuracy":[],
       'mean squared error':[],
       'model':[]}


# ## Logistic Regression

# **Extension of linear regression that’s used for classification tasks, meaning the output variable is binary (e.g., only black or white) rather than continuous (e.g., an infinite list of potential colors)**

# In[331]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[332]:


lr = LogisticRegression()
lr.fit(x_train , y_train)
y_pred = lr.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('LogisticRegression')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax)
plt.show()


# In[333]:


parameters = [{'penalty':['l1','l2']}, 
              {'C':np.logspace(-4, 4, 10)}]
grid_search = GridSearchCV(estimator = lr,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


best_clf_grid=grid_search.fit(x_train, y_train)  
print(best_clf_grid.best_estimator_)


# In[334]:


lr = LogisticRegression(C=166.8)
lr.fit(x_train , y_train)
y_pred = lr.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('LogisticRegression GS')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='afmhot')
plt.show()


# In[335]:


cv_score = cross_val_score(lr, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## Decision Tree

# **Decision Tree is a tree-like graph where sorting starts from the root node to the leaf node until the target is achieved. 
# It is the most popular one for decision and classification based on supervised algorithms. It is constructed by recursive partitioning where each node acts as a test case for some attributes and each edge, deriving from the node, is a possible answer in the test case. Both the root and leaf nodes are two entities of the algorithm.**

# In[304]:


from sklearn.tree import DecisionTreeClassifier


# In[336]:


model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('DecisionTreeClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='plasma')
plt.show()


# In[337]:


cv_score = cross_val_score(model, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## Random Forest

# **Classification or regression model that improves the accuracy of a simple decision tree by generating multiple decision trees and taking a majority vote of them to predict the output, which is a continuous variable (eg, age) for a regression problem and a discrete variable (eg, either black, white, or red) for classification.The random forest algorithm is simple to use and an effective algorithm. It can predict with high accuracy, and that’s why it is very popular.Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.**

# In[338]:


model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('RandomForestClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='YlOrBr')
plt.show()


# In[339]:


cv_score = cross_val_score(model, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## KNN

# **K Nearest Neighbours is a basic algorithm that stores all the available and predicts the classification of unlabelled data based on a similarity measure. In linear geometry when two parameters are plotted on the 2D Cartesian system, we identify the similarity measure by calculating the distance between the points. The same applies here, KNN algorithm works on the assumption that similar things exist in close proximity, simply we can put into the same things stay close to each other.**

# In[340]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('KNeighborsClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='hot')
plt.show()


# In[341]:


cv_score = cross_val_score(classifier, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## XGBoost

# **The XGBoost is having a tree learning algorithm as well as linear model learning, and because of that, it is able to do parallel computation on a single machine. This makes it 10 times faster than any of the existing gradient boosting algorithms. it has become the "state-of-the-art” machine learning algorithm to deal with structured data.**

# In[342]:


model = XGBClassifier(nthread=-1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('XGBClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='Greys')
plt.show()


# In[343]:


cv_score = cross_val_score(model, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## Voting

# **Voting classifiers are ensemble of many classifiers. In voting classifiers we aggregate the predictions of each classifier and predict the class that gets the most votes. This majority vote classifier is called a hard voting classifier. Voting classifiers often achieves a higher accuracy than the best classifier in the ensemble**
# 
# **Types:**
# - Hard Voting 
# - Soft Voting

# ### Hard Voting

# In[344]:


from sklearn.ensemble import  AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier

estimators = [('lr',LogisticRegression()),
              ('dt',DecisionTreeClassifier()),
              ('rf',RandomForestClassifier()),
              ('ab',AdaBoostClassifier()),
              ('xg',XGBClassifier())]
model = VotingClassifier(estimators=estimators,voting='hard')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('HardVotingClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='BuPu')
plt.show()


# ### predict_proba is not available when voting='hard'

# ### Soft Voting

# In[345]:


estimators = [('lr',LogisticRegression()),
              ('dt',DecisionTreeClassifier()),
              ('rf',RandomForestClassifier()),
              ('ab',AdaBoostClassifier()),
              ('xg',XGBClassifier())]
model = VotingClassifier(estimators=estimators,voting='soft')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
men=mean_squared_error(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
compa['accuracy'].append(accuracy)
compa['mean squared error'].append(men)
compa['model'].append('SoftVotingClassifier')                      
print("accuracy score",accuracy)
print("mean squared error :",men)
print(classification_report(y_test, y_pred ,target_names=['Non Pulsar','Pulsar']))
confusion_matrix=metrics.confusion_matrix(y_pred,y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
fig, ax = plt.subplots(figsize=(6,5))
ax.grid(False)
cm_display.plot(ax=ax,cmap='cividis')
plt.show()


# In[346]:


cv_score = cross_val_score(model, x, y, cv=10, scoring='roc_auc')
print("=== All AUC Scores ===")
print(cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score : ", cv_score.mean())

y_score = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# ## compersion between models

# In[350]:


pd.DataFrame(compa,columns=['model','accuracy','mean squared error'])


# In[ ]:




