#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning

# ### Bike Sharing Study

# In[1]:


# Files


# Readme.txt
# hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
# day.csv - bike sharing counts aggregated on daily basis. Records: 731 days


# Dataset characteristics

# Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv


# season : season (1:springer, 2:summer, 3:fall, 4:winter)
# yr : year (0: 2011, 1:2012)
# mnth : month ( 1 to 12)
# hr : hour (0 to 23)

#    holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)

#    workingday : if day is neither weekend nor holiday is 1, otherwise is 0.

#    weathersit : 
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

# temp : Normalized temperature in Celsius. The values are divided to 41 (max)
# atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
# hum: Normalized humidity. The values are divided to 100 (max)
# windspeed: Normalized wind speed. The values are divided to 67 (max)
# casual: count of casual users
# registered: count of registered users


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')


# In[4]:


day_df


# In[5]:


day_df.info()


# In[6]:


day_df.shape


# In[7]:


hour_df


# In[8]:


hour_df.info()


# In[9]:


hour_df.shape


# ### Preprocessing - Data Exploration

# In[11]:


# Renaming columns names to more readable names

day_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)


# In[12]:


# Renaming columns names to more readable names
hour_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)


# In[13]:


# Creating a feature for weekend

hour_df.loc[(hour_df['weekday'] < 6) & (hour_df['weekday'] > 0), 'is_weekend'] = 'No'

hour_df.loc[(hour_df['weekday'] == 0) | (hour_df['weekday'] == 6), 'is_weekend'] = 'Yes'


# In[14]:


# Creating a feature for weekend

day_df.loc[(day_df['weekday'] < 6) & (day_df['weekday'] > 0), 'is_weekend'] = 'No'

day_df.loc[(day_df['weekday'] == 0) | (day_df['weekday'] == 6), 'is_weekend'] = 'Yes'


# In[15]:


# Defining holiday - Yes or No

hour_df.loc[hour_df['is_holiday'] == 0, 'is_holiday'] = 'No'

hour_df.loc[hour_df['is_holiday'] == 1, 'is_holiday'] = 'Yes'


# In[16]:


# Defining holiday - Yes or No

day_df.loc[day_df['is_holiday'] == 0, 'is_holiday'] = 'No'

day_df.loc[day_df['is_holiday'] == 1, 'is_holiday'] = 'Yes'


# In[17]:


# Chaning season names

hour_df.loc[hour_df['season'] == 1, 'season'] = 'Winter'
hour_df.loc[hour_df['season'] == 2, 'season'] = 'Spring'
hour_df.loc[hour_df['season'] == 3, 'season'] = 'Summer'
hour_df.loc[hour_df['season'] == 4, 'season'] = 'Fall'


# In[18]:


# Chaning season names

day_df.loc[hour_df['season'] == 1, 'season'] = 'Winter'
day_df.loc[hour_df['season'] == 2, 'season'] = 'Spring'
day_df.loc[hour_df['season'] == 3, 'season'] = 'Summer'
day_df.loc[hour_df['season'] == 4, 'season'] = 'Fall'


# In[19]:


day_df.head()


# In[20]:


hour_df.head()


# In[21]:


# Dealinh with missing values - using isnull() function  

hour_df.isnull()


# In[233]:


#check for any missing values

hour_df.isnull().values.any()


# In[22]:


# Dealinh with missing values - using isnull() function  

day_df.isnull()


# In[234]:


#check for any missing values

day_df.isnull().values.any()


# In[148]:


# The attribute dteday would require type conversion from object (or string type) to timestamp.

  # That is, 'dteday' should be a 'datetime' object.


# In[149]:


pd.to_datetime(hour_df.datetime)


# In[150]:


pd.to_datetime(day_df.datetime)


# In[151]:


# Setting proper data types - HOUR DF

# date time conversion

hour_df['datetime'] = pd.to_datetime(hour_df.datetime)


# In[152]:


# Setting proper data types - DAY DF

# date time conversion

day_df['datetime'] = pd.to_datetime(day_df.datetime)


# In[153]:


hour_df.info()


# In[29]:


day_df.info()


# In[ ]:





# ### Dealing with Temperature 

# In[32]:


hour_df.temp = hour_df.temp*47 - 8 # Converting temperature to Celsius


# In[33]:


hour_df.temp.hist(bins= 50);


# In[34]:


day_df.temp = day_df.temp*47 - 8 # Converting temperature to Celsius


# In[35]:


day_df.temp.hist(bins= 50);


# In[36]:


### Now it's time to convert the teparute into levels


# In[37]:


hour_df.loc[hour_df['temp'] <= 10, 'temp_level']  = 'low'


# In[38]:


hour_df.loc[(hour_df['temp'] < 20) & (hour_df['temp'] > 10), 'temp_level']  = 'medium'


# In[39]:


hour_df.loc[(hour_df['temp'] >= 20) & (hour_df['temp'] < 25), 'temp_level']  = 'warm'


# In[40]:


hour_df.loc[hour_df['temp'] > 25, 'temp_level']  = 'high'


# In[ ]:





# In[41]:


day_df.loc[day_df['temp'] <= 10, 'temp_level']  = 'low'


# In[42]:


day_df.loc[(day_df['temp'] < 20) & (day_df['temp'] > 10), 'temp_level']  = 'medium'


# In[43]:


day_df.loc[(day_df['temp'] >= 20) & (day_df['temp'] < 25), 'temp_level']  = 'warm'


# In[44]:


day_df.loc[day_df['temp'] > 25, 'temp_level']  = 'high'


# In[ ]:





# ### Creating a feature to joint Registered and Casual users = variable 

# In[45]:


hour_df.columns


# In[47]:


hour_df.head()


# In[49]:


df_hour = pd.melt(hour_df, id_vars=[ 'datetime', 'season', 'year', 'month', 'hour', 'is_holiday',
       'weekday', 'is_workingday', 'weather_condition', 'temp', 'atemp',
       'humidity', 'windspeed', 'total_count',
       'is_weekend', 'temp_level'], value_vars=["casual", "registered"]).sort_values(by = ['datetime', 'hour']).reset_index(drop = True)


# In[50]:


df_hour


# In[52]:


df_day = pd.melt(day_df, id_vars=[ 'datetime', 'season', 'year', 'month', 'is_holiday',
       'weekday', 'is_workingday', 'weather_condition', 'temp', 'atemp',
       'humidity', 'windspeed', 'total_count',
       'is_weekend', 'temp_level'], value_vars=["casual", "registered"]).sort_values(by = ['datetime', 'datetime']).reset_index(drop = True)


# In[53]:


df_day


# ### Dealing with Outliers 

# In[54]:


### Outlier Analysis

import seaborn as sns
import matplotlib.pyplot as plt


# In[55]:


fig, axes = plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(15, 15)
sns.boxplot(data=hour_df,y='total_count',orient="v",ax=axes[0][0])
sns.boxplot(data=hour_df,y='total_count',x="month",orient="v",ax=axes[0][1])
sns.boxplot(data=hour_df,y='total_count',x="weather_condition",orient="v",ax=axes[1][0])
sns.boxplot(data=hour_df,y='total_count',x="is_workingday",orient="v",ax=axes[1][1])
sns.boxplot(data=hour_df,y='total_count',x="hour",orient="v",ax=axes[2][0])
sns.boxplot(data=hour_df,y='total_count',x="temp_level",orient="v",ax=axes[2][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Month', ylabel='Count',title="Box Plot On Count Across Months")
axes[1][0].set(xlabel='Weather Situation', ylabel='Count',title="Box Plot On Count Across Weather Situations")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
axes[2][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[2][1].set(xlabel='temp_level', ylabel='Count',title="Box Plot On Count Across Temperature Levels")


# In[56]:


### Interpretation: 

# The working day and holiday box plots indicate that more bicycles are rent
# during normal working days than on weekends or holidays. 

# The hourly box plots show a local maximum at 8 am and one at 5 pm which indicates
# that most users of the bicycle rental service use the bikes to get to work or school. 
    
# Another important factor seems to be the temperature. 
    
#Higher temperatures lead to an increasing number of bike rents.

#Lower temperatures not only decrease the average number of rents but also shows more outliers in the data.


### ---> We will change the temperature for group levels to see it better 


# In[57]:


sns.histplot(hour_df.total_count);


# In[106]:


sns.histplot(day_df.total_count, bins = 100);


# In[121]:


q1 = hour_df.total_count.quantile(0.25)
print(q1)
q3 = hour_df.total_count.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr)
print(upper_bound)
upper_bound = q3 +(1.5 * iqr)
hour_df = hour_df.loc[(hour_df.total_count >= lower_bound) & (hour_df.total_count <= upper_bound)]


# In[122]:


Q1 = day_df.total_count.quantile(0.25)
Q3 = day_df.total_count.quantile(0.75)
iqr = Q3 - Q1
lower_bound = Q1 -(1.5 * iqr)
upper_bound = Q3 +(1.5 * iqr)
day_df = day_df.loc[(day_df.total_count >= lower_bound) & (day_df.total_count <= upper_bound)]


# In[123]:


sns.histplot(hour_df.total_count);


# In[124]:


hour_df.total_count.max()


# In[125]:


sns.histplot(hour_df.total_count, bins = 100);


# In[126]:


sns.histplot(day_df.total_count);


# In[127]:


day_df.total_count.max()


# In[120]:


sns.histplot(_df.total_count, bins = 100);


# In[ ]:





# ### Visualization

# In[61]:


sns.catplot(x = 'temp_level', y = 'total_count', data = hour_df);


# In[62]:


sns.catplot(x = 'temp_level', y = 'total_count', data = day_df);


# In[63]:


sns.set(rc={'figure.figsize':(11,8)})

sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

#sns.set(style="ticks", context="talk")

#plt.style.use("dark_background")

sns.set_style("dark")

#fig, ax = plt.subplots()

ax = sns.pointplot(data = hour_df[['hour','total_count','season']], x = 'hour', y = 'total_count',
              scale = 0.2, hue = 'season');

plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

ax.set(title = 'Season wise hourly distribution of bike rentals',ylabel= 'mean(total_count)');


# In[64]:


sns.set(rc={'figure.figsize':(11,8)})

sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

#sns.set(style="ticks", context="talk")

#plt.style.use("dark_background")

sns.set_style("dark")

#fig, ax = plt.subplots()

ax = sns.pointplot(data = hour_df[['hour','total_count','is_holiday']], x = 'hour', y = 'total_count',
              scale = 0.2, hue = 'is_holiday');

plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

ax.set(title = 'Holiday wise hourly distribution of bike rentals',ylabel= 'mean(total_count)');


# In[65]:


sns.set(rc={'figure.figsize':(11,8)})

sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

#sns.set(style="ticks", context="talk")

#plt.style.use("dark_background")

sns.set_style("dark")

#fig, ax = plt.subplots()

ax = sns.pointplot(data = hour_df[['hour','total_count','is_weekend']], x = 'hour', y = 'total_count',
              scale = 0.2, hue = 'is_weekend');

plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

ax.set(title = 'Weekend wise hourly distribution of bike rentals',ylabel= 'mean(total_count)');


# In[210]:


# workingday = 0 

sns.set(rc={'figure.figsize':(11,8)})

sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

#sns.set(style="ticks", context="talk")

#plt.style.use("dark_background")

sns.set_style("dark")

#fig, ax = plt.subplots()

ax = sns.pointplot(data = hour_df[['hour','total_count','is_workingday']], x = 'hour', y = 'total_count',
              scale = 0.2, hue = 'is_workingday');

plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

ax.set(title = 'Workingday wise hourly distribution of bike rentals',ylabel= 'mean(total_count)');


# In[66]:


fig,ax = plt.subplots()

sns.barplot(data=hour_df[['month',
                           'total_count']],
              x='month',
              y='total_count',
              ax=ax)

ax.set(title="Monthly distribution of counts")


# In[67]:


fig,ax = plt.subplots()

sns.barplot(data=day_df[['month',
                           'total_count']],
              x='month',
              y='total_count',
              ax=ax)

ax.set(title="Monthly distribution of counts")


# In[68]:


fig,ax = plt.subplots()

sns.barplot(data=hour_df[['season',
                           'total_count']],
              x='season',
              y='total_count',
              ax=ax)

ax.set(title="Seasonal distribution of counts")


# In[69]:


fig,ax = plt.subplots()

sns.barplot(data=day_df[['season',
                           'total_count']],
              x='season',
              y='total_count',
              ax=ax)

ax.set(title="Seasonal distribution of counts")


# In[70]:


fig,ax = plt.subplots()
sns.pointplot(data=hour_df[['hour',
                           'total_count',
                           'weekday']],
              x='hour',
              y='total_count',
              hue='weekday',
              ax=ax)
ax.set(title="Weekday wise hourly distribution of counts")


# In[212]:


# datetime', 'season', 'year', 'month', 'is_holiday',
 #       'weekday', 'is_workingday', 'weather_condition', 'temp', 'atemp',
  #       'humidity', 'windspeed', 'total_count'

corrMatt = hour_df[['temp',
                    'atemp', 
                    'is_holiday',
                    'weekday',
                    'is_workingday',
                    'weather_condition',
                    'humidity', 
                    'windspeed', 
                    'temp_level',
                    'season', 
                    'month',
                    'casual', 
                    'hour',
                    'registered', 
                    'total_count']].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)


# In[214]:


corrMatt = day_df[['temp',
                    'atemp', 
                    'is_holiday',
                    'weekday',
                    'is_workingday',
                    'weather_condition',
                    'humidity', 
                    'windspeed', 
                    'temp_level',
                    'season', 
                    'month',
                    'casual', 
                    'registered', 
                    'total_count']].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)


# In[73]:


plt.figure(figsize=(6,6))
hour_df.groupby(["season"])["total_count"].sum().plot.pie()


# In[74]:


plt.figure(figsize=(6,6))
day_df.groupby(["season"])["total_count"].sum().plot.pie()


# In[76]:


plt.figure(figsize=(5,5))
hour_df.groupby(["weather_condition"])["total_count"].sum().plot.pie()


# In[75]:


plt.figure(figsize=(5,5))
day_df.groupby(["weather_condition"])["total_count"].sum().plot.pie()


# In[77]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(30,30))

columns = ['temp', 'atemp', 'humidity', 'windspeed', 'total_count']

ax= axes.flatten()

idx = 0
for i, val1 in enumerate(columns):
  for j, val2 in enumerate(columns):
    if val1!=val2:
        ax[idx].scatter(hour_df[val1], hour_df[val2])
        ax[idx].set_xlabel(val1)
        ax[idx].set_ylabel(val2)
        idx+=1

plt.show()


# In[79]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(30,30))

columns = ['temp', 'atemp', 'humidity', 'windspeed', 'total_count']

ax= axes.flatten()

idx = 0
for i, val1 in enumerate(columns):
  for j, val2 in enumerate(columns):
    if val1!=val2:
        ax[idx].scatter(day_df[val1], day_df[val2])
        ax[idx].set_xlabel(val1)
        ax[idx].set_ylabel(val2)
        idx+=1

plt.show()


# In[80]:


hour_df.atemp = hour_df.atemp*47 - 8 # Converting A.temperature to Celsius


# In[81]:


plt.figure(figsize= (14,5))

plt.subplot(1,2,1)
hour_df.groupby("atemp")["total_count"].sum().plot(color="r")
plt.title("Total counts")

plt.subplot(1,2,2)
hour_df.groupby("atemp")["total_count"].mean().plot(color="b")
plt.title("Average counts")


# In[274]:


plt.figure(figsize= (14,5))

plt.subplot(1,2,1)
hour_df.groupby("weather_condition")["total_count"].sum().plot(color="r")
plt.title("Total counts")

plt.subplot(1,2,2)
hour_df.groupby("weather_condition")["total_count"].mean().plot(color="b")
plt.title("Average counts")


# In[275]:


plt.figure(figsize= (14,5))

plt.subplot(1,2,1)
hour_df.groupby("temp_level")["total_count"].sum().plot(color="r")
plt.title("Total counts")

plt.subplot(1,2,2)
hour_df.groupby("temp_level")["total_count"].mean().plot(color="b")
plt.title("Average counts")


# In[90]:


day_df.atemp = day_df.atemp*47 - 8 # Converting A.temperature to Celsius


# In[91]:


plt.figure(figsize= (14,5))

plt.subplot(1,2,1)
day_df.groupby("atemp")["total_count"].sum().plot(color="r")
plt.title("Total counts")

plt.subplot(1,2,2)
day_df.groupby("atemp")["total_count"].mean().plot(color="b")
plt.title("Average counts")


# In[84]:


sns.lineplot(x = 'hour', y = 'value', data = df_hour, hue = 'variable');


# In[216]:


sns.lineplot(x = 'weekday', y = 'value', data = df_day, hue = 'variable');


# In[200]:


sns.lineplot(x = 'weekday', y = 'value', data = df_hour, hue = 'variable');


# In[201]:


sns.lineplot(x = 'weekday', y = 'value', data = df_day, hue = 'variable');


# In[ ]:





# In[ ]:


sns.lineplot(x = '', y = 'total_count', data = df_hour, hue = 'variable');


# In[85]:


df_hour_month = df_hour.groupby(['variable', 'month'])['value'].sum().reset_index()
df_hour_month


# In[86]:


sns.lineplot(x = 'month', y = 'value', data = df_hour_month, hue = 'variable');


# In[87]:


sns.catplot(x = 'is_weekend', y = 'value', data = df_hour, hue = 'variable');


# In[240]:


sns.catplot(x = 'hour', y = 'total_count', data = df_hour, hue = 'temp_level');


# In[217]:


sns.catplot(x = 'hour', y = 'value', data = df_hour, hue = 'variable');


# In[ ]:





# ### Adding Pollution dataset

# In[92]:


pollution_df = pd.read_csv('pollution.csv')


# In[93]:


pollution_df.head()


# In[197]:


pollution_df.info()


# In[245]:


pollution_df1 = pd.read_csv('day-pollution.csv')


# In[246]:


pollution_df1.head()


# In[ ]:





# In[254]:


pollution_df1.loc[(pollution_df1['aqi_reading'] >= 0) & (pollution_df1['aqi_reading'] < 50), 'aqi_level']  = 'good'# 0 - 50
pollution_df1.loc[(pollution_df1['aqi_reading'] >= 51) & (pollution_df1['aqi_reading'] < 100), 'aqi_level']  = 'moderate'# 51-100
pollution_df1.loc[(pollution_df1['aqi_reading'] >= 101) & (pollution_df1['aqi_reading'] < 150), 'aqi_level'] = 'unhealthy for sensitive'# 101-150
pollution_df1.loc[(pollution_df1['aqi_reading'] >= 151) & (pollution_df1['aqi_reading'] < 200), 'aqi_level'] = 'unhealthy' # 151-200
pollution_df1.loc[pollution_df1['aqi_reading'] > 201, 'aqi_level']  = 'very unhealthy'# 201-300
                        


# In[255]:


pollution_df1.head()


# In[288]:


# Total_bikes by Pollution - Line Plot

sns.catplot(x = 'aqi_level', y = 'cnt', data = pollution_df1);


# In[298]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = pollution_df1.cnt,
             x = pollution_df1.mnth,
             hue = pollution_df1.aqi_level,
             palette = 'turbo',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['good', 'moderate', 'unhealthy', 'unhealthy for sensitive', 'very unhealthy']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
plt.tight_layout()


# In[ ]:





# ### Working with the Peak Hours - Visualization

# In[88]:


# MONDAY - FRIDAY     # Weekend - No peak/rush

# AM Rush
# 5am - 9:30am        # https://www.wmata.com/fares/basic.cfm

# PM Rush
# 3pm - 7pm


# In[89]:


# District of Columbia Court - Hours of Operation - https://www.dcd.uscourts.gov

# Monday - Friday
# 9:00 a.m. to 4:00 p.m.

# In general, federal building hours of operation are from 7:30 a.m. to 5:00 p.m.

# https://www.gsa.gov/about-us/regions/welcome-to-the-national-capital-region-11/buildings-and-facilities/visiting-or-working-in-federal-buildings


# In[94]:


hour_df.head()


# In[140]:


#import unittest

from nose.tools import *

import time

import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt


# In[141]:


hour_df_non_weekend = hour_df[hour_df['is_weekend'] == 0]
hour_df_non_weekend = hour_df_non_weekend.drop(['is_holiday','is_weekend'],axis=1)
hour_df_is_weekend = hour_df[hour_df['is_weekend'] == 1]
hour_df_is_weekend = hour_df_is_weekend.drop(['is_holiday','is_weekend'],axis=1)
hour_df_is_holiday = hour_df[hour_df['is_holiday'] == 1]
hour_df_is_holiday = hour_df_is_holiday.drop(['is_holiday','is_weekend'],axis=1)
hour_df_non_holiday = hour_df[hour_df['is_holiday'] == 0]
hour_df_non_holiday = hour_df_non_holiday.drop(['is_holiday','is_weekend'],axis=1)


# In[154]:


hour_df.head()


# In[157]:


def hourly_plot(hour_df,title):
   

    """
    Function for plotting bike shares by hour.
    input: 
    df - pandas dataframe 
    title - main title of the plot
    
    """
    assert_true('datetime' in set(hour_df.columns))
    assert_true('total_count' in set(hour_df.columns))
    hour_df.groupby(by=hour_df.datetime.dt.hour)['total_count'].mean().plot()
    plt.title(title)
    plt.ylabel('datetime')
    plt.legend(['workday','is_weekend','is_holiday'],loc=2, fontsize = 'medium')
    
hourly_plot(hour_df_non_weekend, 'Bike shares by hour')
hourly_plot(hour_df_is_weekend, 'Bike shares by hour')
hourly_plot(hour_df_is_holiday, 'Bike shares by hour')


# ### Bike sharing utilization over the two years

# In[159]:




plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.datetime,
             y = hour_df.total_count,
             color = 'steelblue')
plt.tight_layout()
plt.title('Bike sharing utilization over the two years')


# In[ ]:


# Based on the two years dataset, 
 # it seems that the utilization of the bike sharing service has increased over the period. 
  # The number of bikes rented per day also seems to vary depending on the season, 
   # with Spring and Summer months being showing a higher utilization of the service.


# ### Total_bikes by Month - Line Plot

# In[161]:


# Total_bikes by Month - Line Plot

plt.figure(figsize=(15,5))
g = sns.lineplot(x = hour_df.month,
             y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.tight_layout()
plt.title('Bike sharing utilization by Month')


# ### Total_bikes by Month - Box Plot

# In[164]:


# Total_bikes by Month - Box Plot

plt.figure(figsize=(15,5))
sns.boxplot(x = hour_df.month,
            y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.title('Total count of bikes by Month - Box Plot')


# ### Total_bikes by Hour - Line Plot

# In[165]:


# Total_bikes by Hour - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.hour,
             y = hour_df.total_count,
             color = 'steelblue')
plt.xticks([0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
plt.tight_layout()
plt.title('Total count of bikes by Hour - Line Plot')


# In[278]:


# Total_bikes by Temp Level - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.temp_level,
             y = hour_df.total_count,
             color = 'steelblue')

plt.tight_layout()
plt.title('Total count of bikes by Temperature Level - Line Plot')


# In[ ]:


# MONDAY - FRIDAY     # Weekend - No peak/rush

# AM Rush
# 5am - 9:30am        # https://www.wmata.com/fares/basic.cfm

# PM Rush
# 3pm - 7pm

# District of Columbia Court - Hours of Operation - https://www.dcd.uscourts.gov

# Monday - Friday
# 9:00 a.m. to 4:00 p.m.

# In general, federal building hours of operation are from 7:30 a.m. to 5:00 p.m.


# ### Total_bikes by Weekday - Line Plot

# In[166]:


# Total_bikes by Weekday - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.weekday,
             y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.xticks([0,1,2,3,4,5,6])
plt.tight_layout()
plt.title('Total count of bikes by Weekday - Line Plot')


# ### Total_bikes by Holiday - Line Plot

# In[167]:


# Total_bikes by Holiday - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.is_holiday,
             y = hour_df.total_count,
             color = 'steelblue') \
plt.tight_layout()
plt.title('Total count of bikes by Holiday - Line Plot')


# ### Total_bikes by Holidays - Box Plot

# In[170]:


# Total_bikes by Holidays - Box Plot

plt.figure(figsize=(15,5))
sns.boxplot(x = hour_df.is_holiday,
             y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Normal Day', 'Holiday'])
plt.tight_layout()


# In[ ]:


# Utilization of bikes during holidays seems lower and with less peaks


# #### second quartile is the median !!!

# ### Total_bikes by Weekend - Line Plot

# In[168]:


# Total_bikes by Weekend - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.is_weekend,
             y = hour_df.total_count,
             color = 'steelblue') \
plt.tight_layout()
plt.title('Total count of bikes by Weekend - Line Plot')


# ### Total_bikes by Working Day - Box Plot

# In[183]:


# Total_bikes by Working Day - Box Plot

plt.figure(figsize=(15,5))
sns.boxplot(x = hour_df.is_workingday,
             y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Non Working Day', 'Working Day'])
plt.tight_layout()
plt.title('Total count of bikes by Workingday - Box Plot')


# In[ ]:


# Utilization seems higher during working days, with higher peaks.


# ### Total_bikes by Weather Condition - Line Plot

# In[184]:


# Total_bikes by Weather Condition - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.weather_condition,
             y = hour_df.total_count,
             color = 'steelblue') \
   .axes.set_xticklabels(['Clear', 'Mist', 'Light Rain', 'Heavy Rain'])
plt.xticks([1,2,3,4])
plt.tight_layout()
plt.title('Total count of bikes by Weather Condition - Line Plot')


# ### Total_bikes by Actual Temperature - Line Plot

# In[185]:


# Total_bikes by Actual Temperature - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.atemp,
             y = hour_df.total_count,
             color = 'steelblue')
plt.tight_layout()
plt.title('Total count of bikes by Actual Temperature - Line Plot')


# ### Total_Bikes by Hour with Holiday Hue

# In[ ]:


# Total_Bikes by Hour with Holiday Hue


# In[186]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.total_count,
             x = hour_df.hour,
             hue = hour_df.is_holiday,
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Normal Day', 'Holiday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
plt.title('Total count of bikes by Hour with Holiday Hue')


# ### Total_Bikes by Hour with Workingday Hue

# In[ ]:


# Total_Bikes by Hour with Workingday Hue


# In[187]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.total_count,
             x = hour_df.hour,
             hue = hour_df.is_workingday,
             palette = 'inferno',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Non Working Day', 'Working Day']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
plt.tight_layout()
plt.title('Total count of bikes by Hour with Workingday Hue')


# ### Total_Bike by Hour with Weekend Hue

# In[195]:


# Total_Bikes by Hour with Weekend Hue


# In[189]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.total_count,
             x = hour_df.hour,
             hue = hour_df.is_weekend,
             palette = 'inferno',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Weekday', 'Weekend']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
plt.tight_layout()
plt.title('Total count of bikes by Hour with Weekend Hue')


# ### Total_Bikes by Hour with Weekday Hue

# In[ ]:


# Total count of bikes by Hour with Weekday Hue


# In[192]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.total_count,
             x = hour_df.hour,
             hue = hour_df.weekday,
             palette = 'turbo',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
# plt.axhline(hours_df.total_bikes.mean(), color='steelblue')
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
plt.title('Total count of bikes by Hour with Weekday Hue')


# In[ ]:


# The utilization by hour during weekdays differs from the utilization during weekends. 
 # During weekdays, two (2) peaks are present during commute times (around 8am and 5-6pm),while during weekends, utilization is higher during the day between 10am and 6pm.


# ### Total_Bikes by Hour with Weekday Hue for Registered Users

# In[ ]:


# Total_Bikes by Hour with Weekday Hue for Registered Users


# In[193]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.registered,
             x = hour_df.hour,
             hue = hour_df.weekday,
             palette = 'turbo',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
plt.title('Total count of bikes by Hour with Weekday Hue for Registered Users')


# In[ ]:


# Registered users seem to be responsible for the two (2) peaks during commute times. They still use the bikes during the weekends.


# ### Total_Bike by Hour with Weekday Hue for casual Users

# In[194]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.casual,
             x = hour_df.hour,
             hue = hour_df.weekday,
             palette = 'turbo',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
plt.title('Total count of bikes by Hour with Weekday Hue for casual Users')


# In[ ]:


# Casual users are mainly using the bikes during the weekends.


# In[265]:


hour_df.head()


# In[264]:


# weathersit : 
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 


# In[273]:


plt.figure(figsize=(15,5))
g = sns.pointplot(y = hour_df.total_count,
             x = hour_df.hour,
             hue = hour_df.weather_condition,
             palette = 'turbo',
             markers='.',
             errwidth = 1.5)
plt.title('Total count of bikes by Hour with Weather condition Hue for total Users')


# In[ ]:


# Superstorm Sandy – October 29-30, 2012
# Hurricane Sandy came ashore in New Jersey, but the so called “Frankenstorm” had much wider impacts, 
# with damaging winds throughout the DC area along with heavy rain. 
# 32 inches of snow was recorded at Snowshoe Mountain in West Virginia. 
# Meanwhile, high surf took out half of the pier in Ocean City, MD.


# In[270]:


df.loc['2012-10-28 00:01:00' : '2012-10-31 23:00:00']['cnt'].plot(kind = 'line');


# In[272]:


df.loc['2011-01-29 00:01:00' : '2011-01-29 23:00:00']['cnt'].plot(kind = 'line');


# In[286]:


df.loc['2011-03-29 00:01:00' : '2011-03-29 23:00:00']['cnt'].plot(kind = 'line');


# In[ ]:


# Commuteageddon – January 26, 2011
# On the evening of January 26th, a potent area of low pressure moved through the Mid Atlantic, 
# with precipitation starting in the form of rain and even thunderstorms, 
# quickly turning over to heavy wet snow as temperatures plummeted. 
# This corresponded with the evening commute, which many in this area will remember for the rest of their lives.
# Many people were stranded in their cars through the night and many more abandoned their vehicles all together.
# Only 5” of snow was recorded at Reagan National and 7.3” at Dulles Airport, 
# but this all came in the span of a few hours, absolutely crippling the roadways across the region.


# In[269]:


df.loc['2011-01-25 00:01:00' : '2011-01-27 23:00:00']['cnt'].plot(kind = 'line');


# In[271]:


df.loc['2011-01-26 00:01:00' : '2011-01-26 23:00:00']['cnt'].plot(kind = 'line');


# In[279]:


fig,ax = plt.subplots()
sns.pointplot(data=hour_df[['hour',
                           'total_count',
                           'season']],
              x='hour',
              y='total_count',
              hue='season',
              ax=ax)
ax.set(title="Season wise hourly distribution of counts")


# In[284]:


# Total_bikes by atemp - Line Plot

plt.figure(figsize=(15,5))
sns.lineplot(x = hour_df.atemp,
             y = hour_df.total_count,
             color = 'steelblue')

plt.tight_layout()
plt.title('Total count of bikes by Feeling Temperature in Celsius- Line Plot')


# In[ ]:


# weathersit : 
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 


# In[282]:



fig,ax = plt.subplots()
sns.pointplot(data=hour_df[['hour',
                           'total_count',
                           'weather_condition']],
              x='hour',
              y='total_count',
              hue='weather_condition',
              ax=ax)
ax.set(title="Weather condition wise hourly distribution of counts")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Time Series

# In[256]:


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})


# In[257]:


df = pd.read_csv('hour.csv')
#df = pd.read_csv('/content/day.csv')
df.hr = df.hr.astype('str')

df['hr']=df['hr'].apply(lambda x: '{0:0>2}'.format(x))
df.loc[:,'timestamp'] = df['dteday'] + ' ' + df['hr']
df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%Y-%m-%d %H')
df['timestamp']


# In[258]:


df.index=df.timestamp


# In[259]:


df.head()


# In[ ]:





# In[260]:


# Draw Plot

def plot_df(ser, x, y, title="", xlabel='hr', ylabel='cnt', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(ser, x=ser.index, y=ser.cnt, title='Hourly total count bike users from 2011/01 to 2013/01.')    


# In[250]:


df.loc['2012-08-27 00:01:00' : '2012-08-28 23:00:00']['cnt'].plot(kind = 'line');


# In[ ]:




