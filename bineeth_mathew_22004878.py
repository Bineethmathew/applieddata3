# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:46:00 2023

@author: Bineeth Mathew
"""
'''
Importing all Required libraries
used sklear for Kmeans cluter and normalization of the data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt



def dataFrame(file_name, col, value1,years):
    """
    Reading manipulating file with country name
    and returning a dataframe and transpose of the dataframe as return
    """
    # Reading Data for dataframe
    df = pd.read_csv(file_name, skiprows = 4)
    # Grouping data with col value
    df1 = df.groupby(col, group_keys = True)
    #retriving the data with the all the group element
    df1 = df1.get_group(value1)
    #Reseting the index of the dataframe
    df1 = df1.reset_index()
    #Storing the column data in a variable
    a = df1['Country Name']
    # cropping the data from dataframe
    df1 = df1.iloc[35:150,years]
    #df1 = df1.drop(columns=['Indicator Name', 'Indicator Code'])
    df1.insert(loc=0, column='Country Name', value=a)
    #Dropping the NAN values from dataframe Column wise
    df1= df1.dropna(axis = 0)
    #transposing the index of the dataframe
    df2 = df1.set_index('Country Name').T
    #returning the normal dataframe and transposed dataframe
    return df1, df2


# countries which are using for data analysis
years= [35,36,37,38,39,40]
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
popu_c, popu_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "Population, total",
                                       years)
#Printing value by countries
print(popu_c)

#Printing value by Year
print(popu_y)

#returns a numpy array as x
x = popu_c.iloc[:,1:].values

'''
Function to normalize the dataframe
i have used preprocessing MinMaxScaler
'''


def normalizing(value):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

'''
function to find the no of cluster needed by elbow method
the elbow cluster will be used to find the no of clusters which needed to be created
'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(normalized_df,10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(15,7))
plt.plot(range(1, 10), k)
plt.title('The elbow method Using urban population')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)


'''
Ploting Kmeans clusters
5 cluster has been plotted with the reference of the elbow method
'''

plt.figure(figsize=(15,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')


#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')

plt.legend()
# Title of the  plot
plt.title('Clusters of urban population of of 113 countries for year 1990 to 1994')
plt.xlabel('Countries')
plt.ylabel('Population')

plt.show()

popu_c['lables']=lables
print('dataframe with cluster lables', popu_c)
popu_c.to_csv('total population data with cluster label.csv')

# countries which are using for data analysis
years= [35,36,37,38,39,40]
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
popu_c, popu_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "CO2 emissions (kg per PPP $ of GDP)",
                                       years)
#Printing value by countries
print(popu_c)

#Printing value by Year
print(popu_y)

#returns a numpy array as x
x = popu_c.iloc[:,1:].values

'''
Function to normalize the dataframe
i have used preprocessing MinMaxScaler
'''


def normalizing(value):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

'''
function to find the no of cluster needed by elbow method
the elbow cluster will be used to find the no of clusters which needed to be created
'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(normalized_df,10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(15,7))
plt.plot(range(1, 10), k)
plt.title('The elbow method Using CO2 emission')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)


'''
Ploting Kmeans clusters
5 cluster has been plotted with the reference of the elbow method
'''

plt.figure(figsize=(15,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')


#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')
plt.legend()
# Title of the  plot
plt.title('Clusters of CO2 emission of of 113 countries for year 1990 to 1994')
plt.xlabel('Countries')
plt.ylabel('CO2 emissions')

plt.show()

# countries which are using for data analysis
years= [35,36,37,38,39,40]
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
popu_c, popu_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "Electricity production from oil sources (% of total)",
                                       years)
#Printing value by countries
print(popu_c)

#Printing value by Year
print(popu_y)

#returns a numpy array as x
x = popu_c.iloc[:,1:].values

'''
Function to normalize the dataframe
i have used preprocessing MinMaxScaler
'''


def normalizing(value):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

'''
function to find the no of cluster needed by elbow method
the elbow cluster will be used to find the no of clusters which needed to be created
'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(normalized_df,10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(15,7))
plt.plot(range(1, 10), k)
plt.title('The elbow method Using urban population')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)


'''
Ploting Kmeans clusters
5 cluster has been plotted with the reference of the elbow method
'''

plt.figure(figsize=(15,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')


#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')

plt.legend()
# Title of the  plot
plt.title('Clusters of Electricity production from oil sources (% of total) of 113 countries for year 1990 to 1994')
plt.xlabel('Countries')
plt.ylabel('Electricity production')

plt.show()

# countries which are using for data analysis
years= [35,36,37,38,39,40]
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
popu_c, popu_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                                       "Indicator Name", "Access to electricity (% of population)",
                                       years)
#Printing value by countries
print(popu_c)

#Printing value by Year
print(popu_y)

#returns a numpy array as x
x = popu_c.iloc[:,1:].values

'''
Function to normalize the dataframe
i have used preprocessing MinMaxScaler
'''


def normalizing(value):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

'''
function to find the no of cluster needed by elbow method
the elbow cluster will be used to find the no of clusters which needed to be created
'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(normalized_df,10)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(15,7))
plt.plot(range(1, 10), k)
plt.title('The elbow method Using urban population')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)


'''
Ploting Kmeans clusters
5 cluster has been plotted with the reference of the elbow method
'''

plt.figure(figsize=(15,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')


#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')
plt.savefig("urban.png")
plt.legend()
# Title of the  plot
plt.title('Clusters of Access to electricity (% of population)of 113 countries for year 1990 to 1994')
plt.xlabel('Countries')
plt.ylabel('Access to electricity ')
plt.savefig("Access to electricity (% of population) .jpg")
plt.show()
'''
calling dataFrame functions for all the data frame which will be used for curve fitting
we have taken school enrollment of primal and secondary
'''
years= [35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
gdp_c, gdp_y = dataFrame("API_19_DS2_en_csv_v2_4700503.csv",
                         "Indicator Name", "Foreign direct investment, net inflows (% of GDP)",years)
gdp_y['mean']=gdp_y.mean(axis=1)
gdp_y['years'] = gdp_y.index

print(gdp_y)

#Ploting the data of the indian school enrolment for primar and secondary
ax = gdp_y.plot(x = 'years', y = 'mean', figsize=(13, 7), title='Mean gdp of 88 country ', xlabel='Years', ylabel= 'N')

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1999.0
    f = n0 * np.exp(g*t)
    return f


print(type(gdp_y["years"].iloc[1]))
gdp_y["years"] = pd.to_numeric(gdp_y["years"])
print(type(gdp_y["years"].iloc[1]))
#calling exponential function
param, covar = opt.curve_fit(exponential, gdp_y["years"], gdp_y["mean"],
                             p0=(73233967692.102798, 0.03))

gdp_y["fit"] = exponential(gdp_y["years"], *param)

gdp_y.plot("years", ["mean", "fit"],
           title='Data fitting with n0 * np.exp(g*t)',
           figsize=(13, 7))
plt.show()
print(gdp_y)

'''
function for logistic fit which will be used for prediction of students enrolled
before and after the available years
'''
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


#fitting logistic fit
param, covar = opt.curve_fit(logistic, gdp_y["years"], gdp_y["mean"],
                             p0=(3e12, 0.03, 1990.0), maxfev=5000)

sigma = np.sqrt(np.diag(covar))
igma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
gdp_y["logistic function fit"] = logistic(gdp_y["years"], *param)
gdp_y.plot("years", ["mean", "fit"],
           title='Data fitting with f = n0 / (1 + np.exp(-g*(t - t0)))',
           figsize=(7, 7))
plt.show()

#predicting years
year = np.arange(1960, 2010)
print(year)
forecast = logistic(year, *param)
print('forecast=',forecast)

'''
Visualizing  the values of the student enrolment from your 1960 to 2030 with plot
'''
plt.figure()
plt.plot(gdp_y["years"], gdp_y["mean"], label="GDP")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP/year")
plt.legend()
plt.title('Prediction of GDP from 1960 to 2010')

plt.show()

import err_ranges as err
low, up = err.err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(gdp_y["years"], gdp_y["mean"], label="GDP")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")

plt.legend()
plt.show()