import numpy as np
import csv
from scipy import stats, signal
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt


# Read predicted_hr values from CSV file
with open('Predicted HR Values.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    predicted_hr = [float(row[0]) for row in reader]
    window_size = 50
    smoothed_predicted_hr = gaussian_filter(predicted_hr, sigma=10)

    
# Read real_hr values from CSV file
with open('Real HR Values.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    real_hr = [float(row[0]) for row in reader]

# Interpolate the values of the shorter array
interpolated_real_hr = np.interp(np.linspace(0, 1, len(smoothed_predicted_hr)), np.linspace(0, 1, len(real_hr)), real_hr)
time = np.linspace(0, 378, len(smoothed_predicted_hr)) / 60  # Convert to minutes

# Plot the heart rate values
plt.plot(time, smoothed_predicted_hr, label='Predicted HR')
plt.plot(time, interpolated_real_hr, label='Real HR')
plt.xlabel('Time (minutes)')
plt.ylabel('Heart Rate (beats per minute)')
plt.legend()

# Set the y-axis limits
plt.ylim(60, 120)

# Perform ANOVA
fvalue, pvalue = stats.f_oneway(predicted_hr, interpolated_real_hr)
print(len(smoothed_predicted_hr))
print(len(interpolated_real_hr))
print(smoothed_predicted_hr[1])
print(interpolated_real_hr[1])
# Calculate correlation coefficient
correlation_coefficient = np.corrcoef(smoothed_predicted_hr, interpolated_real_hr)[0, 1]
print("Correlation Coefficient:", correlation_coefficient)
print("ANOVA Results:")
print("F-value:", fvalue)
print("p-value:", pvalue)

plt.show()