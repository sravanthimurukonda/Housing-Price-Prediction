#!/usr/bin/env python
# coding: utf-8

# # HOUSING PRICE PREDICTION

# * Housing price prediction is a field of study and application that involves the use of data analysis and predictive modeling to forecast the future values of residential properties. This predictive process is crucial for various stakeholders, including homeowners, real estate professionals, investors, and policymakers, as it enables them to make informed decisions in the dynamic and often volatile real estate market.
# * This dataset provides key features for predicting house prices, including area, bedrooms, bathrooms, stories, amenities like air conditioning and parking, and information on furnishing status. It enables analysis and modelling to understand the factors impacting house prices and develop accurate predictions in real estate markets.
# * With data preprocessing and advanced analytical techniques, this study aims to derive meaningful insights. The preprocessing phase involves essential steps to ensure data quality, handling missing values, normalizing the attributes, and addressing any outliers that might skew subsequent analysis.
# * This research aims not only to explore the dataset and derive insights but also to underscore the significance of leveraging such methodologies in the realm of ecommerce data analytics.

# In[1]:


#importing all libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#importing the dataset
df=pd.read_csv("housing.csv")


# In[3]:


#shows the dataframe
df


# In[4]:


# Lets just check to confirm that our data is stored as dataframe
type(df)


# In[8]:


#To get column names
df.columns


# In[5]:


# To change the all the column names
df.columns=['Price','Area','Bedrms','Bathrms','Stories','Mainroad','Guestrm','Basement','Hotwater','Aircondition','Parking','Prefarea','Furnisherstatus']
df


# In[6]:


#To check no.of rows and columns in dataset
df.shape


# In[7]:


#To check no.of elements in dataset
df.size


# In[8]:


# To get the columns names 
df.columns


# In[9]:


#to show information of dataset 
df.info()


# In[10]:


#Displaying top 5 data fields of dataset
df.head()


# In[11]:


#Displaying bottom 5 data fields of dataset
df.tail()


# In[12]:


#To check data types of columns
df.dtypes


# In[13]:


df.nunique()


# In[14]:


df["Price"].unique()


# In[15]:


df["Area"].unique()


# In[16]:


df["Bedrms"].unique()


# In[17]:


df["Bathrms"].unique()


# In[18]:


df["Stories"].unique()


# In[19]:


df["Mainroad"].unique()


# In[20]:


df["Guestrm"].unique()


# In[21]:


df["Basement"].unique()


# In[22]:


df["Parking"].unique()


# In[23]:


df["Furnisherstatus"].unique()


# In[24]:


# To check the duplicate data
df.duplicated().sum()


# In[25]:


#To check the null values in each column
df.isnull().sum()


# In[26]:


#To check the null values
df.isnull()


# In[27]:


# To print the only null values records
df[df.isnull().any(axis=1)]


# In[28]:


# to check the notnull() values
df.notnull()


# In[29]:


# To count the not null records?
df.notnull().sum()


# In[30]:


# To print the not null records.
df[df.notnull().all(axis=1)]


# In[31]:


# Is used to check the statistical details
df.describe()


# In[32]:


# To find the max values in the object
df.max()


# In[33]:


# To find the min values in the object
df.min()


# In[34]:


df.mean()


# In[35]:


df.std()


# In[36]:


sns.pairplot(df)
plt.show()


# In[37]:


sns.displot(df["Price"])
plt.show()


# In[38]:


sns.countplot(df['Bedrms'])
plt.show()


# In[39]:


df_s=df.corr()
plt.figure(figsize=(10,5))
sns.heatmap(df_s, annot=True)
plt.show()


# In[40]:


plt.scatter(df["Area"], df["Price"])

# Customize the plot
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend(title='Area Category')

# Show the plot
plt.show()


# In[41]:


df1=df.copy()


# In[42]:


# Create box plots for numerical features
sns.boxplot(x=df1['Price'])
plt.show()


# In[43]:


plt.figure
sns.histplot(df['Parking'], kde=True)
plt.title('House Price Prediction')
plt.xlabel('Parking')
plt.show()


# In[44]:


sns.barplot(x=df['Aircondition'],y=df['Bedrms'],hue=df["Furnisherstatus"])


# In[45]:


sns.boxplot(x = 'Furnisherstatus', y = 'Price', hue = 'Aircondition', data = df)


# ### Decision Tree
# *  'Price' is predicted using a Decision Tree Regressor based on 'Area' of products. 'Area' is the feature (independent) variable and 'Price' is the target (dependent) variable in the model's training on the DataFrame 'df'.
# *  To ensure that the data used for prediction is accurate, the code creates a subset of the DataFrame by eliminating rows where "Area" or "Price" are null. Predicted prices for these valid rows are then calculated by the trained regressor using the given 'Area'.
# *  The final result is a table that shows the product descriptions, along with the Decision Tree Regressor-derived "PredictedPrice" that corresponds to the actual "Price."
# 

# In[46]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Assuming 'df' is your DataFrame
target_variable = 'Price'  # Target variable
feature_columns = ['Area']  # Feature columns

# Model Selection
regressor = DecisionTreeRegressor()
regressor.fit(df[feature_columns], df[target_variable])

# Create a subset of 'df' containing rows where both 'Area' and 'Price' are not null
valid_rows = df.dropna(subset=['Area', 'Price']).copy()

# Use the trained regressor to predict prices for the selected products
valid_rows['PredictedPrice'] = regressor.predict(valid_rows[feature_columns])

# Display the product area, previous price, and predicted price in a table
table = valid_rows[['Area', 'Price', 'PredictedPrice']]
print(table)


# Decision Tree Regressor to predict prices based on product price, utilizing available data to create a table presenting the original and predicted prices for the selected products.
# 

# # K-Means

# * K-means clustering is an unsupervised machine learning algorithm that divides data according to similarity within the data into a fixed number of clusters.Applying the K-means algorithm to the dataset, the features 'Area' and 'Price' are utilized. To guarantee uniformity, the algorithm first scales these features. Iterating through a range of possible cluster counts, it calculates the inertia (within-cluster sum of squared distances) for each possible scenario.
# * The "Elbow Method" is applied to determine the ideal number of clusters, which is defined as the point at which the addition of new clusters no longer significantly reduces inertia. The Elbow Method assumes that there are four clusters when the algorithm is run with a chosen cluster count.
# * Based on how close a data point is to the cluster centroids, each data point is assigned to a cluster. The cluster assignments for each data point are then displayed in the 'cluster' column of the updated DataFrame by the code.
# 

# In[47]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

# Selecting features for clustering
features = df[['Area', 'Price']]

# Standardizing the features
#The 'StandardScaler' normalizes these features to ensure they're on similar scales, as K-means is sensitive to scale differences.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# Suppress the FutureWarning which are unwanted
warnings.filterwarnings("ignore", category=FutureWarning)

# Choosing the number of clusters using the Elbow Method
"""Inertia, in the context of K-means clustering, refers to the within-cluster sum of squares.
It measures the compactness of the clusters.Specifically, it calculates the sum of squared
distances between each data point and its centroid within a cluster.The objective of K-means is to minimize this inertia."""
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
"""
The inertia values for different numbers of clusters are calculated and stored in the list.
This method helps visualize the trade-off between the number of clusters and the within-cluster sum of squares.
The 'elbow point' in the plot represents the optimal number of clusters where adding more clusters doesn’t
significantly reduce the inertia, helping in selecting an appropriate number of clusters for the dataset."""

# Plotting the Elbow Method to determine the number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('No.of clusters')
plt.ylabel('Inertia')
plt.show()


# In[48]:


# Applying K-Means with the chosen number of clusters based on elbow-curve
kmeans = KMeans(n_clusters=7, random_state=40, n_init=10)
df['cluster'] = kmeans.fit_predict(data_scaled)
print(df)


# K-Means Outcome
# * The Elbow Method, applied to determine the optimal number of clusters, revealed a pivotal point at 7 clusters. This indicates a suitable partitioning of the data into distinct groups, potentially representing different customer segments or purchasing behaviors within the housing price prediction.
# * By assigning each data point to a specific cluster, the analysis enables a deeper understanding of customer preferences, behaviors, and transaction patterns. These clusters can be leveraged for targeted marketing strategies, inventory optimization, and personalized customer experiences, tailoring services or products to specific cluster preference

# Anomaly Detection
# * Using Isolation Forest, anomaly detection is a method for locating and isolating odd patterns or outliers in a dataset.
# *	A dataset with data on area, price, and furnisher status is analyzed using the Isolation Forest algorithm to find anomalies. The Isolation Forest model builds a set of binary trees to distinguish between "normal" and anomalous data points.
# *	By looking at the isolation depths of the anomalies, one can identify them. The data entries that are flagged as outliers are then printed by the code to highlight the anomalies that were found.
# *	In addition, the scatter plot shows the Area versus Price graphically, with the points colored to indicate which data are normal and which are anomalous. Based on their anomaly scores, the outliers in the dataset can be found using this visual representation.
# 

# In[49]:


from sklearn.ensemble import IsolationForest

# Filtering out rows with negative Quantity or UnitPrice
cleaned_data = df[(df['Area'] >= 0) & (df['Price'] >= 0)]

# Selecting 'Quantity', 'UnitPrice', and 'Description' columns for anomaly detection
anomaly_data = cleaned_data[['Area', 'Price', 'Furnisherstatus']].copy()

# Fit Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Contamination is the expected proportion of outliers

model.fit(anomaly_data[['Area', 'Price']])  # Fit on numerical columns

# Predicting anomalies
anomalies = model.predict(anomaly_data[['Area', 'Price']])
anomaly_data.loc[:, 'anomaly'] = anomalies

# Accessing the detected anomalies
detected_anomalies = anomaly_data[anomaly_data['anomaly'] == -1]
print("Detected Anomalies:")
print(detected_anomalies[['Area', 'Price', 'Furnisherstatus']])

# Plotting anomalies
plt.figure(figsize=(10, 6))
plt.scatter(anomaly_data['Area'], anomaly_data['Price'], c=anomaly_data['anomaly'], cmap='viridis')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Area')
plt.ylabel('Price')
plt.colorbar(label='Anomaly Score')
plt.show()


# Anomaly Outcome
# * A score of -1 signifies a high degree of isolation, marking the data point as an outlier.
# * After applying the Isolation Forest algorithm to the dataset, focusing on the 'Area' and 'Price' columns and excluding negative values, several anomalies were detected. These anomalies represent unusual or atypical prices within the dataset. 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




