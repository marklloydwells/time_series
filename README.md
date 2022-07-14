# Quant / Data Science Interview Task 

In this task we will simulate an average day-to-day task that might be required of a member of the Data Science team.
In this repo, you should produce an application which performs the following tasks:

1. *Data Analysis*: Analyse a dataset of currency pair prices.
2. *Data Modelling*: Perform a modelling exercise based on the currency pair price data.
3. *Testing*: The above tasks should be coded up using test-driven development (TDD) or similar.

We are aware that most candidates will be working full-time and not be able to dedicate blocks of time to the task, therefore, 
candidates will have <b>one week</b> to complete and submit the task.
Please let us know should you have any questions about any of the tasks.

# What we're looking for

* Clear and comprehensive testing
* Clean, interpretable code and styling
* Use of git to track work

## Data Analysis 

In `tests/data/currency_pair_prices.csv`, you will find a CSV file with currency pair 
prices for a variety of currency pairs from the start of 2018 until the end of 2020. Using 
this data, please complete the following:

1. Perform basic analysis of the data. Where relevant, use visualisations and tables to show your results
2. Calculate the correlations between the various time series 
2. Deal with the missing values in the time series
3. Deal with the outliers in the time series
4. Calculate the log returns 


## Data Modelling

Using the same data as above, we are now looking to predict the day-ahead price of 
EUR/USD using previous EUR/USD prices, and the prices of other currency pairs. Please 
develop a model, or models, to perform this task. 

We are not looking for a perfect model, but we are looking for strong reasoning behind
the usage of a particular model and high-level understanding of model functionality.
Where relevant, document your assumptions and steps.

