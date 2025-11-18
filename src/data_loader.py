"""
Module for loading and saving historical weather data.

This module defines a function to download daily historical weather data from
Meteostat for a given location and timeframe. The retrieved data can be saved
to a file in a specified format and location.
"""

from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from meteostat import Point, Daily, units
from config import EXTRACTION_TIMEFRAME, LATITUDE, LONGITUDE, DATA_PATH_RAW


def data_loader(n_years: int = EXTRACTION_TIMEFRAME, lat: float = LATITUDE,
                long: float = LONGITUDE, save_path: str = Path(DATA_PATH_RAW)) -> None:
    """
    Downloads and saves daily historical weather data for a specified location and timeframe.

    Args:
        n_years (int, optional): Number of years to retrieve historical weather data, ending at the current date.
                                 Must be a positive integer. Defaults to EXTRACTION_TIMEFRAME from config.
        lat (float, optional): Latitude of the location. Defaults to LATITUDE from config.
        long (float, optional): Longitude of the location. Defaults to LONGITUDE from config.
        save_path (str, optional): File path where the data should be saved. Defaults to DEFAULT_SAVE_PATH.

    Returns:
        None

    Raises:
        ValueError: If n_years is not a positive integer.
    """

    # Validate timeframe parameter
    if n_years <= 0:
        raise ValueError("Number of years must be a positive integer.")

    # Initialize location based on latitude and longitude
    location = Point(lat, long)

    # Define the date range based on n_years
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=n_years)

    # Fetch daily weather data for the defined location and date range
    data = Daily(location, start_date, end_date)
    data = data.convert(units.scientific)
    data = data.fetch()

    # Save the data to CSV at the specified path
    save_path.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path / 'weather_raw_data.csv')
        
    return
