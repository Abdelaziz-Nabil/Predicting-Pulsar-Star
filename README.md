## About Dataset 
#### [Problem Statement:](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate)
> **Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter. Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,which treat the candidate data sets as binary classification problems.**

#### Attribute Information:
> Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

### Task
**Given a Pulsar Star’s information, build a machine learning model that can classify the Star.**

### contant
* [Importing the libraries](#importing-the-libraries)
* [Reading the data](#Reading-the-data)
* [Exploring the data](#Exploring-the-data)
* [Figure out Data](#Figure-out-Data)
* [Detect Outliers for Every columns](#Detect-Outliers-for-Every-columns)
* [Scaling and Split the data](#Scaling-and-Split-the-data)
* [balance the data](#balance-the-data)
* [Modeling](#Modeling)
* [compersion between models](#compersion-between-models)
