"""
This module contains utility functions for time series analysis, including:
- Stationarity tests (ADF and KPSS).
- Time series plotting (basic time series and seasonal patterns).
- Autocorrelation and Partial Autocorrelation plotting.
- Time series model evaluation using ARIMA.

Each function is designed to facilitate exploration and modeling of time series data.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, kpss
from datetime import timedelta

def stationary_tests(serie: pd.Series):
    """ 
    Perform Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests 
    to check if a time series is stationary.
    
    Args:
        serie (pd.Series): Time series data to test.

    Example:
        stationary_tests(time_series_data)
    """
    # ADF Test
    print(f"Augmented Dickey-Fuller (ADF) - p-value: {adfuller(serie)[1]:.2%}\n"
          "\tHo: Non-Stationarity \n\t"
          "Ha: Is Stationary\n\t")
    
    # KPSS Test
    print(f"Kwiatkowski-Phillips-Schmidt-Shin (KPSS) - p-value: {kpss(serie)[1]:.2%}\n"
          "\tHo: Is Stationary\n\t"
          "Ha: Non-Stationarity\n\t")
    return

def ts_plot(serie: pd.Series, time_index: pd.Series, figsize=(15, 3)):
    """
    Plot a time series with different colors for each year.
    
    Args:
        serie (pd.Series): The time series data to plot.
        time_index (pd.Series): The time index for the data.
        figsize (tuple): The size of the plot. Default is (15, 3).
        
    Example:
        ts_plot(time_series_data, time_index)
    """
    # Creating a color palette for each year
    color_palette = {year: sns.color_palette("tab20", time_index.dt.year.nunique())[i] for i, year in enumerate(time_index.dt.year.unique())}

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(x=time_index, y=serie, hue=time_index.dt.year, palette=color_palette)
    sns.lineplot(x=time_index, y=serie, hue=time_index.dt.year, palette=color_palette, legend=False)
    
    # Adjust x-ticks for better date visibility
    date_list = pd.date_range(start=time_index.min(), end=time_index.max(), freq='2QS')
    plt.xticks(date_list, date_list.strftime('%b\n%Y'))

    # Additional layout settings
    plt.legend(loc='lower center', ncols=time_index.dt.year.nunique())
    plt.grid(linestyle='--', )
    plt.title(f"Data Time Distribution ({serie.name})")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.show()

def ts_quick_insights(serie: pd.Series, time_index: pd.Series, figsize=(15, 5)):
    """
    Generate quick insights about a time series, including trend and seasonal analysis.
    
    Args:
        serie (pd.Series): The time series data.
        time_index (pd.Series): The time index.
        figsize (tuple): The size of the plot. Default is (15, 5).
        
    Example:
        ts_quick_insights(time_series_data, time_index)
    """
    # Calculate moving average
    moving_average = serie.rolling(window=30, min_periods=1).mean()

    # Linear regression for trend
    x = range(len(serie))
    slope, intercept, r_value, p_value, std_err = linregress(x, serie)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Trend plot
    sns.scatterplot(x=x, y=serie, ax=ax[0])
    sns.lineplot(x=x, y=moving_average, color='orange', ax=ax[0], label='Moving Average')
    sns.lineplot(x=[0, len(serie)], y=[intercept, slope * len(serie) + intercept],
                 color='red', ax=ax[0], label='Linear Regression')
    
    ax[0].set_xlim([-50, len(serie) + 50]) 
    ax[0].set_xticks([i for i, d in enumerate(time_index) if d in pd.date_range(start=time_index.min(), end=time_index.max(), freq='2QS')])
    ax[0].set_xticklabels(pd.date_range(start=time_index.min(), end=time_index.max(), freq='2QS').strftime('%b\n%y'))
    ax[0].grid(linestyle='--')
    ax[0].set_title(f"Trend Analysis with Linear Regression\nSlope: {np.degrees(np.arctan(slope)):.1f}°")

    # Seasonal plot
    for year in time_index.dt.year.unique():
        serie_per_year = serie.loc[time_index.dt.year == year].reset_index(drop=True)
        sns.lineplot(serie_per_year, color='black', alpha=0.3, ax=ax[1])

    # Layout for seasonal plot
    month_ticks = np.cumsum([calendar.monthrange(1997, month)[1] for month in range(1, 13)])
    ax[1].set_xticks(month_ticks, [])
    ax[1].secondary_xaxis(location=0).set_xticks(month_ticks - np.diff(month_ticks, prepend=0) / 2,
                                                 labels=pd.date_range(start='1997-01-01', end='1997-12-31', freq='ME').strftime('%b'))
    ax[1].grid(linestyle='--')
    ax[1].set_title("Visualizing Seasonal Patterns Across Years")
    ax[1].set_xlabel("\nTime Within Year")
    ax[1].set_xlim([-5, 371])

    plt.tight_layout()
    plt.show()

def plot_acf_pacf(serie: pd.Series, focus: str = 'normal'):
    """
    Plot both the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for a given time series.
    
    Args:
        serie (pd.Series): The time series data.
        focus (str): Type of ACF/PACF plot. Can be 'normal', 'seasonal', or 'both'. Default is 'normal'.
        
    Example:
        plot_acf_pacf(time_series_data)
    """
    # Initialize the figure

    if focus == 'normal':
        fig = plt.figure(figsize=(15, 6))
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        plot_acf(serie, lags=32, ax=ax1)
        ax1.set_title('Autocorrelation Month')
        plot_pacf(serie, lags=32, ax=ax2)
        ax2.set_title('Partial Autocorrelation Month')

    elif focus == 'seasonal':
        fig = plt.figure(figsize=(15, 6))
        ax3 = plt.subplot2grid((2, 1), (0, 0))
        ax4 = plt.subplot2grid((2, 1), (1, 0))
        plot_acf(serie, lags=400, ax=ax3)
        ax3.set_title('Autocorrelation Yearly')
        ax3.set_xlim(300, 400)
        plot_pacf(serie, lags=400, ax=ax4)
        ax4.set_title('Partial Autocorrelation Yearly')
        ax4.set_xlim(300, 400)

    else:
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        plot_acf(serie, lags=32, ax=ax1)
        ax1.set_title('Autocorrelation Month')
        plot_pacf(serie, lags=32, ax=ax2)
        ax2.set_title('Partial Autocorrelation Month')
        plot_acf(serie, lags=400, ax=ax3)
        ax3.set_title('Autocorrelation Yearly')
        ax3.set_xlim(300, 400)
        plot_pacf(serie, lags=400, ax=ax4)
        ax4.set_title('Partial Autocorrelation Yearly')
        ax4.set_xlim(300, 400)

    plt.tight_layout()
    plt.show()
    return

def ts_model_eval(data: pd.Series, len_test:int,
                  p:int, d:int, q:int, m:int = 0,
                  P:int = 0,D:int = 0,Q: int = 0,
                  steps: int = 1):
    """
    Evaluate an ARIMA model by splitting the data into training and testing sets, fitting the ARIMA model,
    and evaluating its forecasting performance using RMSE and MAPE.
    
    Args:
        data (pd.Series): Time series data to model.
        len_test (int): Number of observations for testing.
        p (int), d (int), q (int): ARIMA order parameters.
        m (int): Period for seasonal ARIMA. Default is 0.
        P (int), D (int), Q (int): Seasonal ARIMA order parameters. Default is 0.
        steps (int): Number of steps for forecasting. Default is 1.
        
    Example:
        ts_model_eval(time_series_data, len_test=30, p=1, d=1, q=1)
    """
    # Split data
    N = len(data)
    train = data.head(N-len_test)
    test = data.tail(len_test)

    # ARIMA model fitting
    model_base = ARIMA(endog=train, order=(p, d, q),
                       seasonal_order=(P, D, Q, m))
    model_base_fit = model_base.fit()
    display(model_base_fit.summary())

    # Forecasting
    list_forecast_steps = []
    list_intv_forecast = []
    for i in tqdm(range(len_test, 0, -steps), desc=f"Forecast {len_test} days"):
        train_step = data.head(N-i)
        model = ARIMA(endog=train_step, order=(1, 1, 2))
        model_fit = model.fit()
        list_forecast_steps.append(model_fit.forecast(steps=steps))
        list_intv_forecast.append(model_fit.get_forecast(steps=1).conf_int(alpha=0.05))

    # Calculate error
    forecast_steps = pd.concat(list_forecast_steps)
    forecast_interval_steps = pd.concat(list_intv_forecast)

    error = test - forecast_steps
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(abs(error/test))

    # Plot results
    plt.figure(figsize=(15, 4))
    sns.lineplot(train, label='Train')
    sns.lineplot(test, label='Test', color='green')
    plt.fill_between(forecast_interval_steps.index,
                     forecast_interval_steps[f'lower {data.name}'],
                     forecast_interval_steps[f'upper {data.name}'],
                     alpha=0.2)
    sns.lineplot(forecast_steps, label='Forecast', color='orange', linestyle='--')

    plt.xlim(data.index.max() - timedelta(days=int(1.5*len_test)), data.index.max())
    plt.title(f'RMSE: {rmse:.2f} MAPE: {mape:.2%}')
    plt.show()
    return model_base_fit, forecast_steps
