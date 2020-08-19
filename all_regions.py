# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression 

PREDICTOR_DAYS = 20  # количество дней в качестве предикторов
PREDICTED_DAYS = [1, 5, 10, 15, 30]  # на какое количество дней вперед предсказывать данные
PREDICTED_REGION = 'Санкт-Петербург'


def transform_data(df: pd.DataFrame):
    """
    Трансформирует данные для обучения.
    Нужны кумулятивные данные (заражений, смертей, выздоровлений)
    за каждый день.
    """
    # проверка, что за все дни есть данные
    all_days = pd.to_datetime(df['Date']).diff().iloc[1:].dt.days == 1
    assert all(all_days), 'В датах присутствуют пропуски'
    df = df[['Confirmed', 'Deaths', 'Recovered']]
    df['Removed'] = df['Deaths'] + df['Recovered']
    df['Infected'] = df['Confirmed'] - df['Removed']
    df['I_pct_change'] = df['Infected'].pct_change()
    df = df[['Infected', 'I_pct_change']]
    df = df.iloc[1:]  # удаляется первая строчка с NaN в I_pct_change
    # обновим индекс
    df.reset_index(inplace=True, drop=True)
    return df


def make_learning_data(df: 'pd.DataFrame',
                       predictor_days: int,
                       predicted_days: 'iterable') -> 'pd.DataFrame':
    """
    Описание
    """
    stop_iteration = predictor_days + max(predicted_days)
    x_data = []
    y_data = []
    for i in range(len(df) - stop_iteration):
        x_values = df.iloc[i:i+predictor_days]['I_pct_change'].reset_index(drop=True)
        x_values = x_values.to_frame().T
        x_data.append(x_values)
        # pct_change зависимой переменной нужно посчитать относительно последнего предиктора
        x_last_infected = df.iloc[i + predictor_days - 1]['Infected']
        y_infected = {str(day): df.iloc[i + predictor_days - 1 + day]['Infected'] for day in predicted_days}
        y_pct_change = {key: [(val - x_last_infected) / x_last_infected] for key, val in y_infected.items()}
        y_data.append(pd.DataFrame(y_pct_change))
    x_data = pd.concat(x_data, ignore_index=True)
    y_data = pd.concat(y_data)
    return x_data, y_data


# %%
source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'
frame = pd.read_csv(os.path.join(source_folder, file_name))

# %%
# Объединяем данные с разных регионов в один набор
train_x = []
train_y = []
test_x = []
test_y = []
for region, region_frame in frame.groupby('Region/City'):
    transformed = transform_data(region_frame)
    x, y = make_learning_data(transformed, PREDICTOR_DAYS, PREDICTED_DAYS)
    if region != PREDICTED_REGION:
        train_x.append(x)
        train_y.append(y)
    else:
        test_x.append(x)
        test_y.append(y)

train_x = pd.concat(train_x, ignore_index=True)
train_y = pd.concat(train_y, ignore_index=True)

# %%

# %%
region_frame
# %%
