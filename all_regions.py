"""
Модуль для машинного обучения - для предсказания количества
инфицированных в одном регионе на основе обученной модели
на данных других регионов
!!! В данные нужно добавить временные точки(индексом, например)
"""
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
    df['delta_t'] = pd.to_datetime(df['Date']).diff().dt.days
    df['Infected'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    df['I_pct_change'] = df['Infected'].pct_change()
    df = df[['delta_t', 'Region/City', 'Infected', 'I_pct_change']]
    df = df.iloc[1:]  # удаляется первая строчка с NaN в I_pct_change
    # обновим индекс на всякий случай
    df.reset_index(inplace=True, drop=True)
    return df


def make_learning_data(df: 'pd.DataFrame',
                       predictor_days: int,
                       predicted_days: 'iterable') -> 'pd.DataFrame':
    """
    Описание
    """
    x_data = []
    y_data = []
    single_dataset_size = predictor_days + max(predicted_days)
    for i in range(len(df) - single_dataset_size):
        # проверка качества датасетов, плохие не включаем
        dataset = df.iloc[i:i+single_dataset_size]
        error1 = (dataset['delta_t'] != 1).any()
        error2 = dataset['I_pct_change'].isin([np.nan, np.inf, -np.inf]).any()
        if error1 or error2:
            continue

        x_values = df.iloc[i:i+predictor_days]['I_pct_change'].reset_index(drop=True)
        x_values = x_values.to_frame().T
        x_data.append(x_values)
        # pct_change зависимой переменной нужно посчитать относительно последнего предиктора
        x_last_infected = df.iloc[i + predictor_days - 1]['Infected']
        y_infected = {str(day): df.iloc[i + predictor_days - 1 + day]['Infected'] for day in predicted_days}
        y_pct_change = {key: [(val - x_last_infected) / x_last_infected] for key, val in y_infected.items()}
        y_data.append(pd.DataFrame(y_pct_change))
    if x_data:  # возвращаем результат только если он есть
        x_data = pd.concat(x_data, ignore_index=True)
        y_data = pd.concat(y_data)
        return x_data, y_data
    else:
        return None, None


source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'
frame = pd.read_csv(os.path.join(source_folder, file_name))

# Объединяем данные с разных регионов в один набор
train_x = []
train_y = []
test_x = []
test_y = []
for region, region_frame in frame.groupby('Region/City'):
    transformed = transform_data(region_frame)
    x, y = make_learning_data(transformed, PREDICTOR_DAYS, PREDICTED_DAYS)
    if x is not None:
        if region != PREDICTED_REGION:
            train_x.append(x)
            train_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)

train_x = pd.concat(train_x, ignore_index=True)
train_y = pd.concat(train_y, ignore_index=True)

# %%
# Пробуем обычную линейную регрессию

lr_model = LinearRegression().fit(train_x, train_y.iloc[:, 3])
# %%
lr_model.score(train_x, train_y.iloc[:, 3])
# %%
lr_model.predict(test_x[0])
# %%
lr_model.score(test_x[0], test_y[0].iloc[:, 3])
# %%
def predict_by