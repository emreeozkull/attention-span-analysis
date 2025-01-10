import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
from collections import defaultdict
from prophet import Prophet
import numpy as np
from scipy import stats

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

plot_data()

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

def analyze_day_of_week():
    df = pd.DataFrame({'date': dates, 'duration': durations})
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Calculate average duration by day of week
    day_avg = df.groupby('day_of_week')['duration'].agg(['mean', 'count']).reset_index()
    
    # Set specific day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_avg['day_of_week'] = pd.Categorical(day_avg['day_of_week'], categories=day_order, ordered=True)
    day_avg = day_avg.sort_values('day_of_week')
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(day_avg['day_of_week'], day_avg['mean'])
    plt.title('Average Viewing Duration by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Duration (minutes)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\nViewing Statistics by Day of Week:")
    for _, row in day_avg.iterrows():
        print(f"{row['day_of_week']}: {row['mean']:.2f} minutes (Sessions: {row['count']})")

def detect_binge_watching(threshold_hours=3):
    df = pd.DataFrame({'date': dates, 'duration': durations})
    df = df.sort_values('date')
    
    binge_sessions = []
    current_binge = []
    
    for i in range(len(df)-1):
        current_session = df.iloc[i]
        next_session = df.iloc[i+1]
        
        if (next_session['date'] - current_session['date']) <= timedelta(hours=threshold_hours):
            if not current_binge:
                current_binge.append(current_session)
            current_binge.append(next_session)
        else:
            if len(current_binge) > 1:
                binge_sessions.append(current_binge)
            current_binge = []
    
    print(f"\nBinge Watching Sessions (>{threshold_hours}h gap):")
    for binge in binge_sessions:
        total_duration = sum(session['duration'] for session in binge)
        print(f"Date: {binge[0]['date'].strftime('%Y-%m-%d')}, Episodes: {len(binge)}, Total Duration: {total_duration:.2f} minutes")

def analyze_weekly_patterns():
    df = pd.DataFrame({'date': dates, 'duration': durations})
    
    # Get the start of each week
    df['week_start'] = df['date'].dt.to_period('W').dt.start_time
    
    # Group by week start date
    weekly_total = df.groupby('week_start')['duration'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_total['week_start'], weekly_total['duration'], marker='o')
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title('Weekly Total Viewing Time')
    plt.xlabel('Week Starting Date')
    plt.ylabel('Total Duration (minutes)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Call the new analysis functions
analyze_day_of_week()
detect_binge_watching()
analyze_weekly_patterns()

def predict_future_durations(days_to_predict=30):
    # Prepare data for Prophet
    df = pd.DataFrame({'date': dates, 'duration': durations})
    df = df.sort_values('date')
    
    # Calculate daily averages
    daily_avg = df.groupby('date')['duration'].mean().reset_index()
    
    # Rename columns to match Prophet requirements
    prophet_df = daily_avg.rename(columns={'date': 'ds', 'duration': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_df)
    
    # Create future dates dataframe
    future_dates = model.make_future_dataframe(periods=days_to_predict)
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.scatter(prophet_df['ds'], prophet_df['y'], 
               color='blue', alpha=0.5, label='Actual Values')
    
    # Plot predictions
    plt.plot(forecast['ds'], forecast['yhat'], 
            color='red', linewidth=2, label='Predicted Values')
    
    # Plot confidence intervals
    plt.fill_between(forecast['ds'],
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    color='red', alpha=0.1, label='Confidence Interval')
    
    plt.title('Average Viewing Duration Prediction')
    plt.xlabel('Date')
    plt.ylabel('Duration (minutes)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    
    # Print predictions for the next few days
    future_predictions = forecast.tail(days_to_predict)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print("\nPredictions for the next few days:")
    for _, row in future_predictions.iterrows():
        print(f"Date: {row['ds'].strftime('%Y-%m-%d')}")
        print(f"Predicted duration: {row['yhat']:.2f} minutes")
        print(f"Confidence interval: ({row['yhat_lower']:.2f}, {row['yhat_upper']:.2f})")
        print()

    # Calculate model performance metrics
    actual_values = prophet_df['y'].values
    predicted_values = forecast[:len(actual_values)]['yhat'].values
    
    mse = np.mean((actual_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    print("\nModel Performance Metrics:")
    print(f"Root Mean Square Error: {rmse:.2f} minutes")
    print(f"Mean Absolute Error: {mae:.2f} minutes")

# Call the prediction function
predict_future_durations(730)


def analyze_yearly_duration():
    df = pd.DataFrame({'date': dates, 'duration': durations})
    df['year'] = df['date'].dt.year
    
    print("\n=== Yearly Duration Analysis ===\n")
    
    # Calculate yearly statistics
    yearly_stats = df.groupby('year').agg({
        'duration': ['count', 'mean', 'std', 'sum']
    }).round(2)
    yearly_stats.columns = ['count', 'mean_duration', 'std_duration', 'total_duration']
    
    print("Yearly Statistics:")
    print(yearly_stats)
    
    # Create duration categories with fewer bins (3 instead of 4)
    # Using quantile-based binning to ensure equal distribution
    df['duration_category'] = pd.qcut(df['duration'], 
                                    q=3, 
                                    labels=['Short', 'Medium', 'Long'])
    
    # Create contingency table: Year vs Duration Category
    contingency = pd.crosstab(df['year'], df['duration_category'])
    
    # Check if we have enough observations in each cell
    min_expected = 5
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Check if expected frequencies meet minimum requirements
    valid_chi_square = (expected >= min_expected).all()
    
    print("\nContingency Table (Observed Frequencies):")
    print(contingency)
    print("\nExpected Frequencies:")
    expected_df = pd.DataFrame(
        expected, 
        index=contingency.index, 
        columns=contingency.columns
    )
    print(expected_df.round(2))
    
    print("\nChi-Square Test for Independence:")
    print("H0: No relationship between year and viewing duration")
    print("H1: There is a relationship between year and viewing duration")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if valid_chi_square:
        print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Failed to reject'} null hypothesis")
        print(f"{'There is' if p_value < 0.05 else 'There is no'} significant relationship between year and duration")
    else:
        print("\nWarning: Chi-square test may not be valid due to low expected frequencies")
        print("Consider collecting more data or using fewer categories")
    
    # Rest of the visualization code remains the same
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Heatmap of the contingency table
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Distribution of Viewing Duration Categories by Year')
    
    # 2. Yearly average duration
    yearly_stats['mean_duration'].plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Average Viewing Duration by Year')
    ax2.set_ylabel('Minutes')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot to show distribution
    sns.boxplot(data=df, x='year', y='duration', ax=ax3)
    ax3.set_title('Distribution of Viewing Duration by Year')
    ax3.set_ylabel('Duration (minutes)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Mann-Kendall trend test (unchanged as it's working well)
    years = sorted(df['year'].unique())
    if len(years) > 1:
        yearly_means = [df[df['year'] == year]['duration'].mean() for year in years]
        trend_stat, trend_p = stats.kendalltau(years, yearly_means)
        
        print("\nMann-Kendall Trend Test:")
        print("H0: No trend in viewing duration over years")
        print("H1: There is a trend in viewing duration over years")
        print(f"Correlation coefficient: {trend_stat:.4f}")
        print(f"p-value: {trend_p:.4f}")
        print(f"Conclusion: {'Reject' if trend_p < 0.05 else 'Failed to reject'} null hypothesis")
        
        if trend_p < 0.05:
            trend_direction = "decreasing" if trend_stat < 0 else "increasing"
            print(f"There is a significant {trend_direction} trend in viewing duration over years")
        else:
            print("No significant trend detected in viewing duration over years")
        
        # Calculate percentage change
        first_year_mean = df[df['year'] == years[0]]['duration'].mean()
        last_year_mean = df[df['year'] == years[-1]]['duration'].mean()
        percent_change = ((last_year_mean - first_year_mean) / first_year_mean) * 100
        
        print(f"\nPercentage change from {years[0]} to {years[-1]}: {percent_change:.1f}%")

# Call the function
analyze_yearly_duration()

