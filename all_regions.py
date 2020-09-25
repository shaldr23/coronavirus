"""
Модуль для машинного обучения - для предсказания количества
инфицированных в одном регионе на основе обученной модели
на данных других регионов
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
    df.index = pd.to_datetime(df['Date']).dt.date
    df['delta_t'] = pd.to_datetime(df['Date']).diff().dt.days
    df['Infected'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    df['I_pct_change'] = df['Infected'].pct_change()
    df = df[['delta_t', 'Region/City', 'Infected', 'I_pct_change']]
    df = df.iloc[1:]  # удаляется первая строчка с NaN в I_pct_change
    return df


def make_learning_data(df: 'pd.DataFrame',
                       predictor_days: int,
                       predicted_days: 'iterable',
                       test_data=False) -> 'tuple of objects':
    """
    Описание ...
    Если test_data == True, то возвращаются дополнительные данные,
    иначе - пустые списки
    """
    x_data = []
    y_data = []
    y_dates = []
    x_last_infected_data = []
    single_dataset_size = predictor_days + max(predicted_days)
    for i in range(len(df) - single_dataset_size + 1):
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
        # Соберем даты для точек зависимой переменной
        if test_data:
            y_date_values = {str(day): [df.iloc[i + predictor_days - 1 + day].name] for day in predicted_days}
            y_dates.append(pd.DataFrame(y_date_values))
            x_last_infected_data.append(x_last_infected)
    if x_data:  # возвращаем результат только если он есть
        x_data = pd.concat(x_data, ignore_index=True)
        y_data = pd.concat(y_data, ignore_index=True)
        if test_data:
            y_dates = pd.concat(y_dates, ignore_index=True)
            x_last_infected_data = pd.Series(x_last_infected_data)
        return x_data, y_data, y_dates, x_last_infected_data
    else:
        return (None,) * 4


def make_train_test_sets(df: 'pd.DataFrame',
                         last_learning_date: 'YYYY-MM-DD',
                         predictor_days: int,
                         predicted_days: 'iterable'):
    """
    Производит тренировочный и тестовый наборы данных
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_data: {<region_1>: {'x': pd.DataFrame,
                             'y': pd.DataFrame,
                             'y_dates': pd.DataFrame,
                             'x_last_infected': pd.Series},
                ...}
    """
    train_x = []
    train_y = []
    test_data = {}
    # Делаем тренировочный набор
    train_df = df[pd.to_datetime(df['Date']) <= last_learning_date]
    for region, region_frame in train_df.groupby('Region/City'):
        transformed = transform_data(region_frame)
        x, y, *trash = make_learning_data(transformed, predictor_days, predicted_days)
        if x is not None:
            train_x.append(x)
            train_y.append(y)
    train_x = pd.concat(train_x, ignore_index=True)
    train_y = pd.concat(train_y, ignore_index=True)

    # Делаем тестовые наборы
    # Надо, чтобы предикторы были датированы до last_learning_date включительно,
    # А предсказываемые данные находились после last_learning_date.
    # Не забываем, что функция transform_data удаляет первую строчку датасета.
    start_set_date = pd.to_datetime(last_learning_date) - pd.Timedelta(predictor_days, unit='d')
    end_set_date = pd.to_datetime(last_learning_date) + pd.Timedelta(max(predicted_days), unit='d')
    test_df = df[pd.to_datetime(df['Date']).between(start_set_date, end_set_date)]
    for region, region_frame in test_df.groupby('Region/City'):
        transformed = transform_data(region_frame)
        x, y, y_dates, x_last_infected = make_learning_data(transformed, predictor_days,
                                                            predicted_days, test_data=True)
        if x is not None:
            test_data[region] = {}
            test_data[region]['x'] = x
            test_data[region]['y'] = y
            test_data[region]['y_dates'] = y_dates
            test_data[region]['x_last_infected'] = x_last_infected
    return train_x, train_y, test_data


def train_predict(train_x,
                  train_y,
                  test_data,
                  model: 'sklearn model'):
    """
    Description
    """
    points_models = {key: model().fit(train_x, train_y[key]) for key in train_y.columns}
    predictions = {}
    for region, data in test_data.items():
        predicted_points = {key: model.predict(data['x']) for key, model in points_models.items()}
        predicted_points = pd.DataFrame(predicted_points)
        x_last_infected = data['x_last_infected'][0]
        predicted_infected = predicted_points * x_last_infected + x_last_infected
        predictions[region] = {}
        predictions[region]['pct_change'] = predicted_points
        predictions[region]['infected'] = predicted_infected
    return predictions

source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'
frame = pd.read_csv(os.path.join(source_folder, file_name))


train_x, train_y, test_data = make_train_test_sets(frame, '2020-06-01', 10, np.arange(1, 16))
predictions = train_predict(train_x, train_y, test_data, LinearRegression)

# %%
# Исследуем предсказания
from sklearn.ensemble import RandomForestRegressor
predictions = train_predict(train_x, train_y, test_data, RandomForestRegressor)
