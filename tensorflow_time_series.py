from sklearn.metrics import mean_absolute_error
import os
import pandas as  pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
zip_path= tf.keras.utils.get_file(

    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
print(zip_path)

csv_path, _=os.path.splitext(zip_path)

print(csv_path)

df=pd.read_csv(csv_path)
df=df[5::6]
print(df)
# parse the 'Date Time' column
date_info = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')


df = df.loc[:,['T (degC)']]

prediction_periods=30

y_hats=[]

for i in reversed(range(prediction_periods)):
   
    
    h= i+1
    window_inex=(len(df)- h)

    y_win= df[: window_inex].tail(30)

    y_hats.append(y_win.mean())

print(y_hats)
print(df.tail(prediction_periods))

print(mean_absolute_error(df.tail(prediction_periods),y_hats))

# Convert y_hats to a proper list if it's not already
y_hats_values = [val[0] for val in y_hats]  # Adjust based on your y_hats structure

# Since we've kept the date information, we can now use it for plotting
actual_dates = date_info.tail(prediction_periods)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(actual_dates, df['T (degC)'].tail(prediction_periods), label='Actual Temperatures', marker='o', linestyle='-')
plt.plot(actual_dates, y_hats_values, label='Predicted Temperatures', marker='x', linestyle='--')

# Format date on the x-axis using mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))  # Updated format string
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Updated locator

plt.gcf().autofmt_xdate()  # Auto-rotate the dates for better spacing

plt.title('Actual vs Predicted Temperatures Over Time')
plt.xlabel('Date Time')
plt.ylabel('Temperature (degC)')
plt.legend()
plt.grid(True)
plt.show()
