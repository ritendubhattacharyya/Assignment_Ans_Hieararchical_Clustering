## Q1

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("D:\\360DigiTMG\\DataScience\\9. Data mining Unsupervised learning - Hierarchical Clustering\\Submit\\EastWestAirlines.xlsx", sheet_name='data')
df.head()

df.describe()

df.columns
df.drop(['ID#'], axis=1, inplace=True)

plt.hist(df.Balance)
plt.hist(df.Qual_miles)
plt.hist(df.cc1_miles)
plt.hist(df.cc2_miles)
plt.hist(df.cc3_miles)
plt.hist(df.Bonus_miles)
plt.hist(df.Bonus_trans)
plt.hist(df.Flight_miles_12mo)
plt.hist(df.Flight_trans_12)
plt.hist(df.Days_since_enroll)

import seaborn as sn
sn.pairplot(df);

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
normalised_df_array = minmax.fit_transform(df)
normalised_df = pd.DataFrame(normalised_df_array, index=df.index, columns=df.columns)
normalised_df.describe()

plt.hist(normalised_df.Balance)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(normalised_df, method="single", metric="euclidean")

# Dendrogram
plt.figure(figsize=(15,8))
plt.title("Hierarchichal Clustering Dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")

sch.dendrogram(
    z,
    leaf_rotation=0,
    leaf_font_size=10,
    truncate_mode='lastp'
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=2, linkage='single', affinity='euclidean').fit(normalised_df)
clustering.labels_

normalised_df['clust'] = pd.Series(clustering.labels_)
normalised_df.head()
normalised_df1 = normalised_df.loc[:, ['clust', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award?']]

mean_normalised = normalised_df1.iloc[:, 2:].groupby(normalised_df1.clust).mean()

## Q2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

crime_data = pd.read_csv("D:\\360DigiTMG\\DataScience\\9. Data mining Unsupervised learning - Hierarchical Clustering\\Submit\\crime_data.csv")

crime_data.head()

crime_data.drop(['Unnamed: 0'], axis=1, inplace=True)

crime_data.head()

crime_data.describe()

plt.hist(crime_data.Murder)
plt.hist(crime_data.Assault)
plt.hist(crime_data.UrbanPop)
plt.hist(crime_data.Rape)

sn.pairplot(crime_data)

from sklearn.preprocessing import MinMaxScaler
crime_data_minmax = MinMaxScaler()
crime_data_norm_array = crime_data_minmax.fit_transform(crime_data)
crime_data_norm = pd.DataFrame(crime_data_norm_array, index=crime_data.index, columns=crime_data.columns)

crime_data_norm.describe()

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

crime_data_z = linkage(crime_data, method="complete", metric="euclidean")
crime_data_z

plt.title("Crime Data Hierarchy Cluster")
plt.xlabel("Index")
plt.ylabel("Distance")
dendrogram(crime_data_z, leaf_rotation=0, leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering

crime_data_alg = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete").fit(crime_data_norm)
crime_data_alg.labels_

crime_data['clust'] = pd.Series(crime_data_alg.labels_)
crime_data1 = crime_data.loc[:, ['clust', 'Murder', 'Assault', 'UrbanPop', 'Rape']]

crime_data1.iloc[:, 1:].groupby(crime_data.clust).mean()

## Q3 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder

tele_customer_churn = pd.read_excel("D:\\360DigiTMG\\DataScience\\9. Data mining Unsupervised learning - Hierarchical Clustering\\Submit\\Telco_customer_churn.xlsx")
tele_customer_churn.var()
columns_with_var = tele_customer_churn.var() != 0
columns = tele_customer_churn.columns 
for i in columns:
    if(i in columns_with_var.index):
        if(columns_with_var[i] == 0):
            tele_customer_churn.drop(i, axis=1, inplace=True)
            
            
tele_customer_churn.drop('Quarter', axis=1, inplace=True)
tele_customer_churn.dtypes
tele_customer_churn_rearranged = tele_customer_churn.loc[:, ['Customer ID', 'Referred a Friend', 'Offer', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method', 'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']]
tele_customer_churn_rearranged.dtypes

# enc = OneHotEncoder()
# enc_tele_customer_churn = pd.DataFrame(enc.fit_transform(tele_customer_churn_rearranged.iloc[:, 1:18]).toarray())

object_tele_customer_churn_rearranged = tele_customer_churn_rearranged.iloc[:, 1:18] # taking out the string datatype column in another dataframe
tele_customer_churn_rearranged.drop(object_tele_customer_churn_rearranged.columns, axis=1, inplace=True) # dropping the string related columns from actual dataframe except customer id
customer_id = tele_customer_churn_rearranged.iloc[:, 0:1] # taking out the customer id from the actual dataset
tele_customer_churn_rearranged.drop(['Customer ID'], axis=1, inplace=True)

object_tele_customer_churn_rearranged = pd.get_dummies(object_tele_customer_churn_rearranged, drop_first=True) # one hot encoding on that dataframe which consist string data only

minMaxScaler = MinMaxScaler()
normalised_tele_customer_churn_rearranged_array = minMaxScaler.fit_transform(tele_customer_churn_rearranged)
normalised_tele_customer_churn_rearranged = pd.DataFrame(normalised_tele_customer_churn_rearranged_array, index=tele_customer_churn_rearranged.index, columns=tele_customer_churn_rearranged.columns)

final_tele_customer = pd.concat([normalised_tele_customer_churn_rearranged, object_tele_customer_churn_rearranged], axis=1)

z = linkage(final_tele_customer, method='complete', metric="euclidean")

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(z, truncate_mode='lastp')
plt.show()

tele_customer_h = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(final_tele_customer) 
tele_customer_h.labels_

tele_customer_labels = pd.Series(tele_customer_h.labels_)
final_tele_customer['cluster'] = tele_customer_labels

cols = list(final_tele_customer.columns)
cols = [cols[-1]] + cols[:-1]
final_tele_customer = final_tele_customer[cols]

# tele_customer_churn = tele_customer_churn.loc[:, ['cluster', 'Customer ID', 'Referred a Friend', 'Offer', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method', 'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']]

mean_tele_custoemr = final_tele_customer.iloc[:, 1:].groupby(final_tele_customer.cluster).mean()


## Q4
auto_insurance = pd.read_csv("D:\\360DigiTMG\\DataScience\\9. Data mining Unsupervised learning - Hierarchical Clustering\\Submit\\AutoInsurance.csv");
auto_insurance.dtypes
auto_insurance.drop(['Customer', 'State', 'Education', 'Gender', 'Location Code', 'Marital Status', 'Sales Channel', 'Vehicle Class', 'Vehicle Size'], axis=1, inplace=True)

auto_insurance_object = auto_insurance.select_dtypes('object')
auto_insurance_numeric = auto_insurance.select_dtypes(['int64', 'float64'])

auto_insurance_object = pd.get_dummies(auto_insurance_object, drop_first=True)
auto_insurance_enc = MinMaxScaler()
normalised_auto_insurance = pd.DataFrame(auto_insurance_enc.fit_transform(auto_insurance_numeric))
auto_insurance_final = pd.concat([auto_insurance_object, normalised_auto_insurance], axis=1)

z = linkage(auto_insurance_final, method="complete", metric='euclidean')
plt.title("Auto Insurance Hierarchical")
plt.xlabel("Index")
plt.ylabel("Distance")
dendrogram(z, truncate_mode="lastp")
plt.show()

auto_insurance_h = AgglomerativeClustering(n_clusters=3, linkage="complete", affinity="euclidean").fit(auto_insurance_final)

auto_insurance_labels = pd.Series(auto_insurance_h.labels_)
auto_insurance['cluster'] = auto_insurance_labels
auto_insurance[auto_insurance['cluster']==0].head()
auto_insurance[auto_insurance['cluster']==1].head()
auto_insurance[auto_insurance['cluster']==2].head()
