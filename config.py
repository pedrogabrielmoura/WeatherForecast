from pathlib import Path

PROJ_PATH_LIST = ['C:/Users/PedroMoura/Documents/GitHub/WeatherForecast', 
<<<<<<< HEAD
                  'C:/Users/pedro/Documents/GitHub/WeatherForecast']
=======
                  '/home/pedro.moura/Pessoal/WeatherForecast']
>>>>>>> main

for path in PROJ_PATH_LIST:
    if Path(path).exists():
        PROJ_PATH = Path(path)

DATA_PATH_DATA = PROJ_PATH / 'data'
DATA_PATH_RAW = DATA_PATH_DATA / 'raw'
DATA_PATH_WRANGLE = DATA_PATH_DATA / 'wrangle'
NOTEBOOK_PATH = PROJ_PATH / 'notebooks'


# Config for data extractions
EXTRACTION_TIMEFRAME = 10 # Years
LATITUDE = 41.881832
LONGITUDE = -87.623177


