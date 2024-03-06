import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, normaltest

# Load data
df = pd.read_csv('hour.csv')

# Set page title
st.title("Bike Sharing Analysis Dashboard")

# Sidebar
st.sidebar.title("Filter Options")

# Date and Time Selection
min_date = pd.to_datetime(df['dteday'].min()).date()
max_date = pd.to_datetime(df['dteday'].max()).date()
selected_date = st.sidebar.date_input("Select Date", min_value=min_date, max_value=max_date, value=min_date)
selected_hour = st.sidebar.slider("Select Hour", 0, 23)

# Filtering data based on selected date and hour
filtered_data = df[(df['dteday'] == selected_date.strftime('%Y-%m-%d')) & (df['hr'] == selected_hour)]

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Visualization
st.subheader("Visualizations")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Line plot for hourly bike rentals
hourly_rentals = df.groupby('hr')['cnt'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_rentals, x='hr', y='cnt', marker='o', color='skyblue')
plt.title("Average Hourly Bike Rentals")
plt.xlabel("Hour")
plt.ylabel("Count")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on season
season_counts = df.groupby('season')['cnt'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=season_counts, x='season', y='cnt', palette='muted')
plt.title("Total Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on weather situation
weather_counts = df.groupby('weathersit')['cnt'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=weather_counts, x='weathersit', y='cnt', palette='pastel')
plt.title("Total Bike Rentals by Weather Situation")
plt.xlabel("Weather Situation")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Scatter plot for temperature vs bike rentals
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='temp', y='cnt', hue='season', palette='bright')
plt.title("Temperature vs Bike Rentals")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on user type
user_counts = df[['casual', 'registered']].sum().reset_index()
user_counts.columns = ['user_type', 'total_rentals']
plt.figure(figsize=(10, 6))
sns.barplot(data=user_counts, x='user_type', y='total_rentals', palette='dark')
plt.title("Total Bike Rentals by User Type")
plt.xlabel("User Type")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on month
month_counts = df.groupby('mnth')['cnt'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=month_counts, x='mnth', y='cnt', palette='Set3')
plt.title("Total Bike Rentals by Month")
plt.xlabel("Month")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on year
year_counts = df.groupby('yr')['cnt'].sum().reset_index()
year_counts['yr'] = year_counts['yr'].map({0: '2011', 1: '2012'})
plt.figure(figsize=(8, 5))
sns.barplot(data=year_counts, x='yr', y='cnt', palette='Set2')
plt.title("Total Bike Rentals by Year")
plt.xlabel("Year")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on holiday
holiday_counts = df.groupby('holiday')['cnt'].sum().reset_index()
holiday_counts['holiday'] = holiday_counts['holiday'].map({0: 'No', 1: 'Yes'})
plt.figure(figsize=(8, 5))
sns.barplot(data=holiday_counts, x='holiday', y='cnt', palette='Set1')
plt.title("Total Bike Rentals on Holidays")
plt.xlabel("Holiday")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Bar plot for bike rentals based on working day
workingday_counts = df.groupby('workingday')['cnt'].sum().reset_index()
workingday_counts['workingday'] = workingday_counts['workingday'].map({0: 'No', 1: 'Yes'})
plt.figure(figsize=(8, 5))
sns.barplot(data=workingday_counts, x='workingday', y='cnt', palette='Set2')
plt.title("Total Bike Rentals on Working Days")
plt.xlabel("Working Day")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# RFM Analysis
# Recency
last_rental_date = pd.to_datetime(df['dteday'].max())
df['last_rental'] = last_rental_date - pd.to_datetime(df['dteday'])
recency = df.groupby('registered')['last_rental'].min().reset_index()

# Frequency
frequency = df.groupby('registered').size().reset_index(name='frequency')

# Monetary
monetary = df.groupby('registered')['cnt'].sum().reset_index()

# Merge RFM metrics
rfm_df = pd.merge(pd.merge(recency, frequency, on='registered'), monetary, on='registered')

# Visualize RFM metrics
st.subheader("RFM Analysis")
st.write(rfm_df)

# Additional Visualizations and Analysis

# Box plot for bike rentals based on hour
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='hr', y='cnt', palette='Set2')
plt.title("Box Plot of Hourly Bike Rentals")
plt.xlabel("Hour")
plt.ylabel("Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation = df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='red')
plt.title("Correlation Heatmap")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Calculate p-values for correlations
correlation_p_values = df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].apply(lambda x: pearsonr(df['cnt'], x)[1])
st.subheader("Correlation P-Values")
st.write(correlation_p_values)

# Determine normality of data
normality_results = {}
for column in ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']:
    p_value = normaltest(df[column])[1]
    if p_value < 0.05:
        normality_results[column] = "Not Normal"
    else:
        normality_results[column] = "Normal"
st.subheader("Normality of Data")
st.write(normality_results)

# Percentage of rentals on holidays vs working days
holiday_rentals = df.groupby('holiday')['cnt'].sum().reset_index()
total_rentals = df['cnt'].sum()
holiday_rentals['percentage'] = holiday_rentals['cnt'] / total_rentals * 100
holiday_rentals['holiday'] = holiday_rentals['holiday'].map({0: 'No', 1: 'Yes'})

plt.figure(figsize=(8, 5))
sns.barplot(data=holiday_rentals, x='holiday', y='percentage', palette='Set1')
plt.title("Percentage of Rentals on Holidays vs Working Days")
plt.xlabel("Holiday")
plt.ylabel("Percentage of Total Rentals")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
