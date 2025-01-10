# Analyzing Personal Attention Span Through Netflix Viewing Patterns

## Project Description

This project aims to explore how my personal attention span has changed over time by analyzing my Netflix viewing history. Understanding these patterns can provide insights into my engagement levels, content preferences, and factors that influence shifts in attention. Reflecting on this can also help us understand more about how society consume media in today’s digital world.

## Data Source and Collection Methods

The primary data source for this project is my personal Netflix viewing history. Netflix allows users to download their viewing activity, which includes details such as the title of the content watched and the date and time of viewing. To obtain this data:

1. **Data Download**: Accessed my Netflix account settings and downloaded the "Viewing Activity" report, which is provided in CSV format.
2. **Supplementary Data**: To enrich the analysis, additional metadata about each title (e.g., genre, duration, release year) will be gathered using third-party databases like IMDb.
   

## Analysis Techniques

The project will involve several key steps:

### 1. Data Preprocessing and Cleaning

- **Data Formatting**: Convert the raw CSV data into a structured format suitable for analysis.
- **Data Cleaning**: Handle missing values, correct inconsistencies, and ensure accurate timestamps.
- **Data Integration**: Merge supplementary metadata with the viewing history to enhance analysis capabilities.

### 2. Exploratory Data Analysis (EDA)

- **Time-Based Analysis**: Examine viewing patterns over different periods (daily, weekly, monthly) to identify trends and fluctuations in attention span.
- **Session Identification**: Define what constitutes a viewing session and analyze the duration and frequency of sessions.
- **Content Analysis**: Categorize watched content by genre, type (movie or series), and length to understand preferences and how they correlate with attention span.
- **Attention Span Metrics**: Develop metrics to quantify attention span, such as average time spent on a single title before switching.

### 3. Data Visualization

- **Heatmaps**: Create visual representations of viewing activity intensity over time.
- **Line Graphs**: Plot changes in average session duration and number of episodes watched consecutively.
- **Bar Charts**: Illustrate the distribution of genres and types of content consumed over time.

### 4. Machine Learning Models

- **Clustering Analysis**: Use algorithms to group similar viewing sessions and identify patterns in attention span.
- **Time Series Forecasting**: Apply models to predict future changes in attention span based on historical data.
- **Classification Models**: Implement decision trees or logistic regression to classify sessions as high or low attention based on defined features.

## Expected Outcomes

- **Insight into Personal Habits**: I aim to gain a deeper understanding of how my attention span varies with different types of content and over time. Specifically, I want to discover:

  - Preferred content lengths and genres that hold my focus.
  - Patterns of distraction or loss of interest.
  - Personal thresholds for watching sessions.
 
## Final report

- "The full analysis and findings are available on the [GitHub Pages website](https://emreeozkull.github.io/)."
- "For the detailed code and technical steps, visit the [Jupyter Notebook](https://github.com/emreeozkull/attention-span-analysis/blob/e6941a8cbd5190dee30741202dc4ab2c8547974e/netflix_dsa_210.ipynb)."

