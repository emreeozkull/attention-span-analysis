import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

dates = []
durations = []

with open('/Users/emreozkul/Desktop/dsa-project/attention-span-analysis/data/ViewingActivity.csv', 'r') as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["Profile Name"] == "C":
            date = datetime.strptime(row["Start Time"].split(" ")[0], '%Y-%m-%d')
            
            # Convert duration string (e.g., "1:23:45") to minutes
            duration_parts = row["Duration"].split(':')
            duration_minutes = (int(duration_parts[0]) * 60 + 
                             int(duration_parts[1]) +
                             int(duration_parts[2]) / 60)
            
            dates.append(date)
            durations.append(duration_minutes)

def plot_data():
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, durations, marker='o')

    # Customize the plot
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Duration (minutes)')
    plt.title('Viewing Duration Over Time')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

#plot_data()

seassions = len(dates)
sum_durations = sum(durations)
avg_duration = sum_durations / seassions

print("Seassions: ",seassions)
print("Sum Durations: ",sum_durations)
print("Average Duration: ",avg_duration)

# Calculate monthly averages
df = pd.DataFrame({'date': dates, 'duration': durations})

# Extract month and year from dates
df['month_year'] = df['date'].dt.to_period('M')

# Calculate monthly averages
monthly_avg = df.groupby('month_year')['duration'].mean().reset_index()
monthly_avg['month_year'] = monthly_avg['month_year'].dt.to_timestamp()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot monthly averages
plt.plot(monthly_avg['month_year'], monthly_avg['duration'], marker='o', linewidth=2)

# Customize the plot
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Add labels and title
plt.xlabel('Month')
plt.ylabel('Average Duration (minutes)')
plt.title('Monthly Average Viewing Duration')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print monthly statistics
print("\nMonthly Statistics:")
for _, row in monthly_avg.iterrows():
    print(f"{row['month_year'].strftime('%Y-%m')}: {row['duration']:.2f} minutes")

# Calculate rolling average duration over time
def plot_average_duration_over_time():
    # Convert dates and durations to pandas Series for easier calculation
    df = pd.DataFrame({'date': dates, 'duration': durations})
    df = df.sort_values('date')
    
    # Calculate 7-day rolling average
    df['rolling_avg'] = df['duration'].rolling(window=7, min_periods=1).mean()

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot both individual points and rolling average
    plt.scatter(df['date'], df['duration'], alpha=0.4, label='Individual Sessions')
    plt.plot(df['date'], df['rolling_avg'], color='red', linewidth=2, label='7-day Rolling Average')

    # Customize the plot
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Duration (minutes)')
    plt.title('Average Viewing Duration Over Time')
    plt.legend()

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_average_duration_over_time()


    