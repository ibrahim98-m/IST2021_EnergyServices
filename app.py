import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb #Statistics data visualization base on matplotlib"
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from pandas import DataFrame
import time as tm
from functools import reduce
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn import  linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import base64


######################## DATA CLEANING #################################

df_raw_17 = pd.read_csv('IST_North_Tower_2017_Ene_Cons.csv')
df_raw_18 = pd.read_csv('IST_North_Tower_2018_Ene_Cons.csv')
df_raw_meteo = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
df_raw_holiday = pd.read_csv('holiday_17_18_19.csv')

# cuting the empty columns
df_clean_17 = df_raw_17.iloc[:, :2] 
df_clean_18 = df_raw_18.iloc[:, :2]
df_clean_meteo = df_raw_meteo.iloc[:, :9] 
df_clean_holiday = df_raw_holiday.iloc[:, :2] 

df_clean_meteo.rename(columns = {'temp_C': 'Temperature (C)'}, inplace = True)
df_clean_meteo.rename(columns = {'yyyy-mm-dd hh:mm:ss': 'Date'}, inplace = True)
df_clean_18.rename(columns = {'Date_start': 'Date'}, inplace = True)
df_clean_17.rename(columns = {'Date_start': 'Date'}, inplace = True)

df_clean_17 = df_clean_17.set_index ('Date', drop = True)
df_clean_18 = df_clean_18.set_index ('Date', drop = True)
df_clean_meteo = df_clean_meteo.set_index ('Date', drop = True)
df_clean_holiday = df_clean_holiday.set_index ('Date', drop = True)

df_clean_17.to_csv('IST_North Tower_17_Clean.csv', encoding='utf-8', index=True)
df_clean_18.to_csv('IST_North Tower_18_Clean.csv', encoding='utf-8', index=True)
df_clean_meteo.to_csv('IST_meteo_Clean.csv', encoding='utf-8', index=True)
df_clean_holiday.to_csv('IST_holiday_Clean.csv', encoding='utf-8', index=True)

df_sort_kW_17 = df_clean_17.sort_values(by = 'Power_kW', ascending = False)
df_sort_kW_18 = df_clean_18.sort_values(by = 'Power_kW', ascending = False)
df_sort_meteo = df_clean_meteo.sort_values(by = 'Temperature (C)', ascending = False)


Q1 = df_clean_18['Power_kW'].quantile(0.25)
Q3 = df_clean_18['Power_kW'].quantile(0.75)
IQR = Q3 - Q1

df_clean4_18 = df_clean_18[df_clean_18['Power_kW'] >df_clean_18['Power_kW'].quantile(0.25) ]
df_clean4_18 = df_clean_17[df_clean_17['Power_kW'] >df_clean_17['Power_kW'].quantile(0.25) ]


raw_data_17=pd.read_csv('IST_North Tower_17_Clean.csv')
raw_data_18=pd.read_csv('IST_North Tower_18_Clean.csv')
raw_data_meteo_clean = pd.read_csv('IST_meteo_Clean.csv')
raw_data_holiday_clean=pd.read_csv('IST_holiday_Clean.csv')

raw_data_holiday_day = raw_data_holiday_clean
raw_data_holiday_day['Date'] = pd.to_datetime(raw_data_holiday_clean.Date, format='%d.%m.%Y').dt.strftime('%d-%m-%Y')
listHoliday = raw_data_holiday_day['Date'].values.tolist()
#raw_data_holiday_day['Date'] = pd.date_range(raw_data_holiday_day['Date'], freq='1H')

# Data formatting 
raw_data_17['Date'] = pd.to_datetime(raw_data_17.Date, format='%d-%m-%Y %H:%M').dt.strftime('%d-%m-%Y %H:%M')
raw_data_18['Date'] = pd.to_datetime(raw_data_18.Date, format='%d-%m-%Y %H:%M').dt.strftime('%d-%m-%Y %H:%M')

#2017 & 2018 data merged
frames = [raw_data_17, raw_data_18]
df_date_merged = pd.concat(frames)
#print(df_date_merged)

# Data formatting 
raw_data_meteo_clean['Date'] = pd.to_datetime(raw_data_meteo_clean.Date, format='%Y-%m-%d %H:%M:%S')#.dt.strftime('%d-%m-%Y %H:%M')
#raw_data_holiday_clean['Date'] = pd.to_datetime(raw_data_holiday_clean.Date, format='%d.%m.%Y').dt.strftime('%d-%m-%Y %H:%M')


raw_data_meteo_clean['Date'] = raw_data_meteo_clean['Date'].dt.round(freq='1H')

raw_data_meteo_clean = raw_data_meteo_clean.reset_index()
raw_data_meteo_clean['Date'] = raw_data_meteo_clean['Date'].apply(lambda x: x.replace(minute=0, second=0))

raw_data_meteo_clean = raw_data_meteo_clean.drop_duplicates(subset='Date')

raw_data_meteo_clean['Date'] = pd.to_datetime(raw_data_meteo_clean.Date, format='%Y-%m-%d %H:%M:%S').dt.strftime('%d-%m-%Y %H:%M')

#Merging data
dfs = [df_date_merged,raw_data_holiday_clean,raw_data_meteo_clean]
df_total = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), dfs)

raw_data_total1 = df_total.drop (columns = ['index'])
raw_data_total = raw_data_total1.set_index ('Date', drop = True)
raw_data_total.to_csv('raw_data_total.csv', encoding='utf-8', index=True)

#Merging data #project2
df_merged_data = [df_date_merged,raw_data_meteo_clean]
df_total_consumption = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), df_merged_data)
df_total_consumption_clean = df_total_consumption.drop (columns = ['index', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 'rain_day'])

df_total_consumption_clean['Date']= pd.to_datetime (df_total_consumption_clean['Date'])
df_total_consumption_clean['Year'] = df_total_consumption_clean['Date'].dt.strftime('%Y')

df_total_Power_Temp = df_total_consumption_clean.set_index ('Date', drop = True)


df_total_Power_Temp_clean= df_total_Power_Temp.dropna()

df_total_Power_Temp_clean.to_csv('df_total_Power_Temp_clean.csv', encoding='utf-8', index=True)


#Cleaning holiday
df_total['Holiday'] = df_total['Holiday'].fillna(0)


df_total.loc[df_total['Date'].str.contains('|'.join(listHoliday)),'Holiday']=1

# cuting the empty columns
df_total_clean= df_total.dropna()

raw_data_total1 = df_total_clean.drop (columns = ['index'])
raw_data_total = raw_data_total1.set_index ('Date', drop = True)
raw_data_total.to_csv('raw_data_total.csv', encoding='utf-8', index=True)

#print(df_total_clean)
df_total_clean.to_csv('Total_clean.csv', encoding='utf-8', index=True)


################################CLUSTERING########################

df_total_clean_clustering = df_total_clean.drop (columns = ['Holiday', 'index', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 'rain_day'])
#print(df_total_clean_clustering)


df_total_clean_clustering['Date'] = pd.to_datetime(df_total_clean_clustering['Date'])
df_total_clean_clustering['Date Day'] = df_total_clean_clustering['Date'].dt.strftime('%d-%m-%Y')
df_total_clean_clustering['Hour'] = df_total_clean_clustering['Date'].dt.strftime('%H')
df_total_clean_clustering['Week Day'] = df_total_clean_clustering['Date'].dt.weekday


df_total_clean_clustering = df_total_clean_clustering.set_index ('Date Day', drop = True)
cluster_data=df_total_clean_clustering.drop(columns=['Date'])
#print(cluster_data)

df_total_clean_clustering.to_csv('Total_clean_v2.csv', encoding='utf-8', index=True)

model = KMeans(n_clusters=3).fit(cluster_data)
pred = model.labels_
#print(pred)

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]
#plt.plot(Nc,score)
#plt.xlabel('Number of Clusters')
#plt.ylabel('Score')
#plt.title('Elbow Curve')
#plt.show()

cluster_data['Cluster']=pred

#Step 4: Cluster Analysis

ax1=cluster_data.plot.scatter(x='Power_kW',y='Temperature (C)',color=cluster_data['Cluster'])
#plt.savefig('Clust_image3.png', dpi=300, bbox_inches='tight')

ax2=cluster_data.plot.scatter(x='Power_kW',y='Hour',color=cluster_data['Cluster'])
#plt.savefig('Clust_image4.png', dpi=300, bbox_inches='tight')

ax3=cluster_data.plot.scatter(x='Power_kW',y='Week Day',color=cluster_data['Cluster'])
#plt.savefig('Clust_image5.png', dpi=300, bbox_inches='tight')

ax4=cluster_data.plot.scatter(x='Hour',y='Week Day',color=cluster_data['Cluster'])
#ax4=cluster_data.plot.scatter(x='Hour',y='Power_kW',color=cluster_data['Cluster'])

fig = plt.figure()
ax = plt.axes(projection="3d")

#plt.savefig('Clust_image2.png', dpi=300, bbox_inches='tight')


#y_km4 = kmeans.fit_predict(cluster_data)
cluster_0=cluster_data[pred==0]
cluster_1=cluster_data[pred==1]
cluster_2=cluster_data[pred==2]


cluster_0
ax.scatter3D(cluster_0['Hour'], cluster_0['Week Day'],cluster_0['Power_kW'],c='red');
ax.scatter3D(cluster_1['Hour'], cluster_1['Week Day'],cluster_1['Power_kW'],c='blue');
ax.scatter3D(cluster_2['Hour'], cluster_2['Week Day'],cluster_2['Power_kW'],c='green');

#plt.show()
#plt.savefig('Clust_image6.png', dpi=300, bbox_inches='tight')


#Step 5: Identifying daily patterns
df=cluster_data
df=df.drop(columns=['Temperature (C)','Week Day','Cluster'])
df.rename(columns = {'Power_kW': 'Power'}, inplace = True)

#Create a pivot table
df_pivot = df.pivot(columns='Hour')
df_pivot = df_pivot.dropna()
#print (df_pivot)

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    
#plt.plot(n_cluster_list,sillhoute_scores)

kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

#print(df_pivot)

fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
        )
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'
    )
#plt.savefig('Clust_image1.png', dpi=300, bbox_inches='tight')


ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')

############################### FEATURE SELECTION ######################

df_data= pd.read_csv('Total_clean.csv')

df_data['Date']=pd.to_datetime(df_data['Date']) # Convert Date into datetime type
df_data=df_data.set_index(df_data['Date'] , drop = True) # set date as index
df_data['Power-1']=df_data['Power_kW'].shift(1) # Previous hour consumption
df_data['Hour']=df_data.index.hour
df_data=df_data.dropna()

df_data = df_data.drop(columns = ['Unnamed: 0', 'index', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 'rain_day'])
df_data['Week day'] = df_data['Date'].dt.weekday
df_data=df_data.drop(columns=['Date'])
#print(df_data.head())
#print(df_data.dtypes)

# Define input and outputs
X=df_data.values
Y=X[:,2]
X=X[:,[0,1,3,4,5]] 
#print(Y)
#print(X)

#Filter methods
features=SelectKBest(k=2,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression of the features
#print(fit.scores_)
features_results=fit.transform(X)
#print(features_results) # k=2:Power-1 and Hour k=3: Temperature and Power-1 and Hour

#Wrapper methods 
model=LinearRegression() # LinearRegression Model as Estimator
rfe=RFE(model,2)# using 2 features
rfe2=RFE(model,3) # using 3 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)

#print( "Feature Ranking (Linear Model, 2 features): %s" % (fit.ranking_)) # Holiday, Power-1
#print( "Feature Ranking (Linear Model, 3 features): %s" % (fit2.ranking_)) #Holiday, Power-1, Hour 

#Emsemble methods

model = RandomForestRegressor()
model.fit(X, Y)
#print(model.feature_importances_) # hour Power-1 Temp

#Feature Extraction/Engineering

#Log of temperature
df_data['logtemp']=np.log(df_data['Temperature (C)'])

# Weekday square
df_data['day2']=np.square(df_data['Week day'])
df_data.head()

#Holiday/weekday
df_data['Holtimesweek']=df_data['Week day']*df_data['Holiday']

#df_data['Power-1']=df_data['Power (kW) [Y]'].shift(1) # Previous hour consumption
#df_data=df_data.dropna()

df_data['HDH']=np.maximum(0,df_data['Temperature (C)']-16)
#print(df_data)

model = RandomForestRegressor()
model.fit(X, Y)
#print(model.feature_importances_) # Power-1  Hour

df_model=df_data.drop(columns=['logtemp','day2','Holtimesweek'])
df_model.to_csv('North_Tower_Total_Hourly_Model.csv', encoding='utf-8', index=True)


########################### REGRESSION #########################################

#Pre processing
df_data=pd.read_csv('North_Tower_Total_Hourly_Model.csv')
df_data['Date'] = pd.to_datetime (df_data['Date']) # create a new column 'data time' of datetime type
df_data['Year'] = df_data['Date'].dt.strftime('%Y')
df_data = df_data.set_index('Date') # make 'datetime' into index

#Split training and test data
X=df_data.values
Y=X[:,2]
X=X[:,[0,1,3,4,5]] 
#feature 2 (month) and feature 5(Energy-2) do not improve significantly
#print(X)

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#print(X_train)
#print(y_train)

###################Linear Regression###############

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_LR[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_LR)
#plt.savefig('linear regression.png', dpi=300, bbox_inches='tight')


#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print("Linear Regression : ", MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)



###################Decision Tree Regressor################

# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print("Decision Tree Regressor : ", MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)

###################### Random forest (best method according to my results)################
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print("Random forest : ", MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_RF[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_RF)
#plt.savefig('random forest.png', dpi=300, bbox_inches='tight')


############################ Extreme Gardient Boosting######################

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)

MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print("Extreme Gardient Boosting : ", MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_XGB[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_XGB)
#plt.savefig('extrem gradient boosting.png', dpi=300, bbox_inches='tight')

######################### Bootstrapping##############

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)

MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print("Bootstrapping : ", MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_XGB[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_XGB)

#################### Neural Networks#################

NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)

MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print("Neural Networks : ", MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_NN[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_NN)
#plt.savefig('neural networks.png', dpi=300, bbox_inches='tight')


###################### Optimized Random forest################
parameters = {'bootstrap': True,
              'min_samples_leaf': 1,
              'n_estimators': 2500, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 500,
              'max_leaf_nodes': 2000}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print("Random forest : ", MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

#plt.plot(y_test[1:200])
#plt.plot(y_pred_RF[1:200])
#plt.show()
#plt.scatter(y_test,y_pred_RF)

################################## DASHBOARD ###########################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('raw_data_total.csv')

df6 = pd.read_csv('data_NT.csv')
available_years = df6['year'].unique()

df1 = pd.read_csv('North_Tower_Total_Hourly_Model.csv')

image_filename = 'linear regression.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 = 'random forest.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())

image_filename3 = 'extrem gradient boosting.png' # replace with your own image
encoded_image3 = base64.b64encode(open(image_filename3, 'rb').read())

image_filename4 = 'neural networks.png' # replace with your own image
encoded_image4 = base64.b64encode(open(image_filename4, 'rb').read())

image_filename5 = 'Clust_image3.png' # replace with your own image
encoded_image5 = base64.b64encode(open(image_filename5, 'rb').read())

image_filename6 = 'Clust_image4.png' # replace with your own image
encoded_image6 = base64.b64encode(open(image_filename6, 'rb').read())

image_filename7 = 'Clust_image5.png' # replace with your own image
encoded_image7 = base64.b64encode(open(image_filename7, 'rb').read())

image_filename8 = 'Clust_image1.png' # replace with your own image
encoded_image8 = base64.b64encode(open(image_filename8, 'rb').read())

image_filename9 = '3D.png' # replace with your own image
encoded_image9 = base64.b64encode(open(image_filename9, 'rb').read())

image_filename10 = 'ist_logo.png' # replace with your own image
encoded_image10 = base64.b64encode(open(image_filename10, 'rb').read())

def generate_page():
        return html.Div(children=[
        html.Div([
            html.H3(children='''
                    Visualization of hourly electricity consumption at North Tower over the last years
                    '''),

            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'x': df['Date'], 'y': df['Power_kW'], 'type': 'line', 'name': 'Power'},
                        {'x': df['Date'], 'y': df['Temperature (C)'], 'type': 'line', 'name': 'Temperature'},
                        
                        ],
                    'layout': {
                        'title': 'North Tower hourly electricity consumption (kWh)'
                        }
                    }
                ),
            ]),

            html.H2(children='   '),

            html.Div([
                html.H3(children='''
                     Visualization of total electricity consumption at North Tower over the last years
                     '''),

            dcc.Graph(
                id='yearly',
                figure={
                    'data': [
                        {'x': df6.year, 'y': df6.NorthTower, 'type': 'bar', 'name': 'North Tower'},
                        #{'x': df6.year, 'y': df6.Total, 'type': 'bar', 'name': 'Total'},
                        ],
                    'layout': {
                        'title': 'North Tower yearly electricity consumption (MWh)'
                        }
                    }
                ),
            
            html.H3(children='Summary Table'),

        dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df6.columns],
                data=df6.to_dict('records'),
              
                )
        
      # html.Table([
       #     html.Thead(
       #         html.Tr([html.Th(col) for col in df6.columns])
        #        ),
         #   html.Tbody([
          #      html.Tr([
           #         html.Td(df6.iloc[i][col]) for col in df6.columns
            #        ]) for i in range(len(df6))
             #   ])
            #])
        
        ])
        ])
        
    

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image10.decode()), style={'height':'20%', 'width':'20%'}),
    html.H1('Project 2 : Dashboard - North Tower data'),
    html.H6('by Ibrahim Minta - ist1100838'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw data', value='tab-1'),
        dcc.Tab(label='Exploratory data analysis', value='tab-2'),
        dcc.Tab(label='Clustering', value='tab-4'),
        dcc.Tab(label='Feature selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-5'),
    ]),
    html.Div(id='tabscontent')
])

@app.callback(Output('tabscontent', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    
    if tab == 'tab-2':
        return html.Div([
            generate_page()
        ]),

    elif tab == 'tab-1':
        return    html.Div([
            html.H3('Raw data'),
            dash_table.DataTable(
                id='tble',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                sort_action='native',
                filter_action='native'
                )
            ])

    elif tab == 'tab-5':
        return    html.Div(children=[
            html.Div([
                html.H4('Linear regression results :'),
                html.H6(' MAE_LR = 3.9590377236004777            MSE_LR = 24.131991734230432            RMSE_LR = 4.912432364341562         '),
                html.H6(' cvRMSE_LR = 0.3009657846101855 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height':'50%', 'width':'50%'})
             ]),
    
            html.Div([
                html.H4('Random forest results :'),
                html.H6(' MAE_LR = 3.1098235844621716            MSE_LR = 16.762044451121458           RMSE_LR = 4.094147585410357      '),
                html.H6('    cvRMSE_LR = 0.25102687365365123 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()), style={'height':'50%', 'width':'50%'})
             ]),
             
            html.Div([
                html.H4('Extrem gradient boosting results :'),
                html.H6(' MAE_LR = 3.198246846110806            MSE_LR = 17.85045840704362           RMSE_LR = 4.2249802848112346          '),
                html.H6('cvRMSE_LR = 0.2590486957343465 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image3.decode()), style={'height':'50%', 'width':'50%'})
           ]),
    
            html.Div([
                html.H4('Neural networks results :'),
                html.H6('MAE_LR = 3.5153580234266792           MSE_LR = 20.15767603107264            RMSE_LR = 4.489730062161047          '),
                html.H6('cvRMSE_LR = 0.2752814542077962 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image4.decode()), style={'height':'50%', 'width':'50%'})
             ])
            
        ])
            
    elif tab == 'tab-4':
        return    html.Div(children=[
            
            html.Div([
                html.H3('Cluster analysis'),
                ]),
                html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image5.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
    
            html.Div([
                html.Div([
                html.H3(' '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image8.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
             
            ], className="row"),
            
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image6.decode()), style={'height':'80%', 'width':'80%'})
           ],className="six columns"),
    
            html.Div([
              html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image7.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
            ], className="row"),
            
            html.Div([
              
                html.Img(src='data:image/png;base64,{}'.format(encoded_image9.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns")
            ])
                
    elif tab == 'tab-3':
        return    html.Div([
            html.H3('The features selected are :'),
            html.H5('- Power_kW'),
            html.H5('- Power-1'),
            html.H5('- Temperature'),
            html.H5('- Hour'),
            html.H5('- Week day'),
            html.H5('- HDH'),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df1.columns],
                data=df1.to_dict('records'),
                sort_action='native',
                filter_action='native'
                )
            ])

if __name__ == '__main__':
    app.run_server(debug=False)