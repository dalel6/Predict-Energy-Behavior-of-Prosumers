import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('./backend')  # Replace '/path/to/app/backend' with the actual path to the 'backend' module

from backend import my_component  # Import the custom component

# Load the CSV data
def load_data():
    data = pd.read_csv("./modeling/production_client10.csv")
    data['datetime'] = pd.to_datetime(data['datetime'])
    return data

data = load_data()

# Function to prepare chart data and options
def prepare_chart_data(data, column_name, start_date, end_date):
    # Filter data for the selected date range
    filtered_data = data[(data['datetime'] >= pd.to_datetime(start_date)) & (data['datetime'] <= pd.to_datetime(end_date))]
    
    # Prepare the data for the chart
    chart_data = filtered_data[['datetime', column_name]]
    chart_data['datetime'] = chart_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return chart_data

# Date input widgets for selecting the start and end date
st.title("Electricity Prices ")
start_date = st.date_input('Select start date', value=data['datetime'].min().date())
end_date = st.date_input('Select end date', value=data['datetime'].max().date())

# Create the Streamlit app for elect_prices
chart_data_elect_prices = prepare_chart_data(data, 'elect_prices', start_date, end_date)

line_chart_option_elect_prices = {
    "xAxis": {
        "type": "category",
        "data": chart_data_elect_prices['datetime'].tolist(),
        "axisLabel": {
            "interval": 2  # Show all labels
        }
    },
    "yAxis": {"type": "value"},
    "series": [{
        "data": chart_data_elect_prices['elect_prices'].tolist(),
        "type": "line"
    }],
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {"type": "cross"}
    }
}

st_echarts(options=line_chart_option_elect_prices, height="400px")

# Create the Streamlit app for lowest and highest prices
st.title("Lowest and Highest GAZ prices")
chart_data_lowest_prices = prepare_chart_data(data, 'lowest_price_per_mwh', start_date, end_date)
chart_data_highest_prices = prepare_chart_data(data, 'highest_price_per_mwh', start_date, end_date)

combined_chart_data = pd.merge(chart_data_lowest_prices, chart_data_highest_prices, on='datetime', suffixes=('_lowest', '_highest'))

line_chart_option_prices = {
    "xAxis": {
        "type": "category",
        "data": combined_chart_data['datetime'].tolist(),
        "axisLabel": {
            "interval": 2  # Show all labels
        }
    },
    "yAxis": {"type": "value"},
    "series": [
        {
            "name": "Lowest Price per MWh",
            "data": combined_chart_data['lowest_price_per_mwh'].tolist(),
            "type": "line",
            "itemStyle": {"color": "blue"},
            "lineStyle": {"color": "blue"}
        },
        {
            "name": "Highest Price per MWh",
            "data": combined_chart_data['highest_price_per_mwh'].tolist(),
            "type": "line",
            "itemStyle": {"color": "red"},
            "lineStyle": {"color": "red"}
        }
    ],
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {"type": "cross"}
    },
    "legend": {
        "data": ["Lowest Price per MWh", "Highest Price per MWh"]
    }
}

st_echarts(options=line_chart_option_prices, height="400px")

# Load model
model = tf.keras.models.load_model('./modeling/modelproduction.h5')

# Load data
df = pd.read_csv('./modeling/production_client10.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Define columns
cols = ['eic_count', 'installed_capacity', 'temperature', 'dewpoint',
        'rain', 'snowfall', 'surface_pressure', 'cloudcover_total', 'windspeed_10m',
        'winddirection_10m', 'shortwave_radiation', 'direct_solar_radiation',
        'diffuse_radiation', 'lowest_price_per_mwh', 'highest_price_per_mwh',
        'elect_prices', 'target']

# Split data into train and test
train_size = int(len(df) * 0.8)
test_df = df[train_size:]

# Prepare test data
df_test = test_df[cols].astype(float)
scaler = StandardScaler()
df_test_scaled = scaler.fit_transform(df_test)

# Define the past and future steps for the model
n_past = 24  # number of past days to consider
n_future = 1  # number of future days to predict

# Prepare testX and testY
testX = []
testY = []
for i in range(n_past, len(df_test_scaled) - n_future + 1):
    testX.append(df_test_scaled[i - n_past:i])
    testY.append(df_test_scaled[i + n_future - 1:i + n_future, -1])  # Assuming the last column is still the target

testX, testY = np.array(testX), np.array(testY)

# Define the number of days for prediction
n_days_for_prediction = 48

# Fit scaler on target for inverse transform later
target_scaler = StandardScaler()
target_scaler.fit(df_test[['target']])

# Predict on the last n_days_for_prediction
prediction_scaled = model.predict(testX[-n_days_for_prediction:])

# Inverse transform predictions to original scale
prediction = target_scaler.inverse_transform(prediction_scaled)

# Prepare datetime for plotting
datetime_for_plotting = test_df['datetime'].iloc[n_past + len(testX) - n_days_for_prediction:n_past + len(testX)]

# Streamlit app
st.title('Model Predictions Over Time')

# Plotting the results
fig, ax = plt.subplots()
ax.plot(datetime_for_plotting, df_test['target'].iloc[n_past + len(testX) - n_days_for_prediction:n_past + len(testX)].values, label='Actual')
ax.plot(datetime_for_plotting, prediction, label='Predicted')
ax.set_xlabel('DateTime')
ax.set_ylabel('Target Value')
ax.legend()

st.pyplot(fig)

# Embed the custom JavaScript component
st.title('Custom JavaScript Component')
my_component()
