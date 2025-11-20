from pathlib import Path

PROJ_PATH_LIST = ['C:/Users/PedroMoura/Documents/GitHub/WeatherForecast', 
                  'C:/Users/pedro/Documents/GitHub/WeatherForecast', 
                  '/home/pedro.moura/Pessoal/WeatherForecast']

for path in PROJ_PATH_LIST:
    if Path(path).exists():
        PROJ_PATH = Path(path)

DATA_PATH_RAW = PROJ_PATH / 'data' / 'raw'
DATA_PATH_WRANGLE = PROJ_PATH / 'data' / 'wrangle'
NOTEBOOK_PATH = PROJ_PATH / 'notebooks'


# Config for data extractions
EXTRACTION_TIMEFRAME = 5 # Years
LATITUDE = 41.881832
LONGITUDE = -87.623177


