# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
# from helpack.timetools import time_it

# ---------- Начальные параметры -------------------------------------

REGION = 'Москва'
TO_SAVE = True  # сохранять ли данные в excel

# ---------- Функции -------------------------------------------------


def gradual_change(seq):
    """
    Функция, заменяющая в последовательности неизменные значения
    на меняющиеся: [1, 2, 2, 2, 3] -> [1, 2, 2.33, 2.66, 3]
    """
    seq = list(seq)
    result = seq[:1]
    processed_vals = []
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            processed_vals.append(seq[i])
            if i == len(seq) - 1:
                result.extend(processed_vals)
        else:
            if processed_vals:
                increment = (seq[i] - processed_vals[0]) / (len(processed_vals) + 1)
                processed_vals = [num * increment + val for num, val in enumerate(processed_vals, 1)]
                result.extend(processed_vals)
                processed_vals = []
            result.append(seq[i])
    return result


def add_stats(df: pd.DataFrame, population, fill_values=True,
              use_gradual_change=False):
    """
    Добавление данных в DataFrame S, I, R, Beta, Gamma.
    В этой ветке функция берет производную в точке как
    (следующее значение - предыдущее значение) / 2 дня.
    Данные для каждого дня последовательности не требуются.
    Нужны лишь кумулятивные данные (заражений, смертей, выздоровлений) и даты.
    Если в какие-то дни количество общих случаев не меняется -
    Используется функция gradual_change.
    Первая строчка данных в итоге удаляется,
    т.к. показателей dI, dR, dS для нее нет.
    """
    df = df[['Date', 'Confirmed', 'Deaths', 'Recovered']]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    if fill_values:
        full_dates = pd.date_range(df['Date'].min(), df['Date'].max())
        full_dates = pd.DataFrame({'Date': full_dates})
        full_dates['Date'] = full_dates['Date'].dt.date
        df = full_dates.merge(df, how='left', on='Date')
        df.fillna(method='ffill', inplace=True)
    if use_gradual_change:
        for col in df.columns:
            df[col] = gradual_change(df[col])
    df['Removed'] = df['Deaths'] + df['Recovered']
    df['Infected'] = df['Confirmed'] - df['Removed']
    df['Suspected'] = population - df['Infected'] - df['Removed']
    df['dI'] = -df['Infected'].diff(-2).shift(1)/2
    df['dR'] = -df['Removed'].diff(-2).shift(1)/2
    dt = pd.to_datetime(df['Date']).diff().dt.days
    df['Gamma'] = df['dR'] / dt / df['Infected']
    df['Beta'] = (df['dI'] / dt + df['Infected'] * df['Gamma']) * population / (df['Infected'] * df['Suspected'])
    df['R_0'] = df['Beta'] / df['Gamma']
    df = df.iloc[1:-1]  # удаляется первая и последняя строчки с NaN в dI, dR, dS
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    return df


def simulate_dynamic(P: 'population', I: 'infected',
                     R: 'removed', beta_func: 'function', gamma_func: 'function',
                     start_time=2, cycles=100) -> dict:
    """
    Function to simulate SIR model. beta and gamma depend on time.
    Return dict, not DataFrame, because it is MUCH faster
    Вводятся начальные данные, они не включаются в симуляцию
    """
    S = P - I - R
    data = {'Suspected': [],
            'Infected': [],
            'Removed': []}
    for time in range(start_time, start_time + cycles):
        beta = beta_func(time)
        gamma = gamma_func(time)
        S = S - beta * I * S / P
        I = I + beta * I * S / P - gamma * I
        R = R + gamma * I
        data['Suspected'].append(S)
        data['Infected'].append(I)
        data['Removed'].append(R)
    return data


def func_neg_exp(x, a, b):
    return a * np.exp(-b * x)


def func_neg_exp2(x, a, b, c):
    return a * np.exp(-b * x + c)


def func_exp(x, a, b, c):
    return a * np.exp(b * x + c)


def func_lin(x, a, b):
    return a * x + b


def func_log(x, a, b, c, d):
    return a * np.log(b * x + c) + d


def func_logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def func_polynom_2p(x, a, b, c):
    return a*x**2 + b*x + c


def simulate_graphics(dataset: pd.DataFrame,
                      population,
                      training_range: '(start_index, end_index)',
                      cycles=100,
                      beta_func=func_neg_exp,
                      gamma_func=func_lin,
                      show_pictures=True,
                      return_result=False):
    """
    Функция, осуществляющая подбор уравнений для Beta и Gamma,
    использующая функцию simulate_dynamic для симуляции заражения с некоторого
    момента времени и строящая графики.
    """
    training_set = dataset.loc[training_range[0]:training_range[1] + 1]
    training_set.reset_index(inplace=True)  # сбрасываем индекс, т.к. эксп. регрессия капризная
    xdata = training_set.index

    y_beta = training_set['Beta']
    beta_opt = curve_fit(beta_func, xdata, y_beta)[0]
    y_gamma = training_set['Gamma']
    gamma_opt = curve_fit(gamma_func, xdata, y_gamma)[0]

    # Строим графики для Beta и Gamma, если нужно:
    if show_pictures:
        # График для Beta
        y_beta_fitted = beta_func(xdata, *beta_opt)
        r2 = r2_score(y_beta, y_beta_fitted)
        plt.plot(xdata, y_beta_fitted, 'r-',
                 label=f'params: {tuple(beta_opt)}')
        plt.plot(xdata, y_beta, label='real')
        plt.legend()
        plt.title(f'Beta: {beta_func.__name__}, R2={r2}')
        plt.show()
        # График для Gamma
        y_gamma_fitted = gamma_func(xdata, *gamma_opt)
        r2 = r2_score(y_gamma, y_gamma_fitted)
        plt.plot(xdata, y_gamma_fitted, 'r-',
                 label=f'params: {tuple(gamma_opt)}')
        plt.plot(xdata, y_gamma, label='real')
        plt.legend()
        plt.title(f'Gamma: {gamma_func.__name__}, R2={r2}')
        plt.show()

    # Симуляция
    init_vals = dataset.loc[training_range[1]]  # From here we take I and R
    start_sim_time = training_set.index[-1] + 1
    simulated = simulate_dynamic(population, init_vals['Infected'], init_vals['Removed'],
                                 beta_func=lambda x: beta_func(x, *beta_opt),
                                 gamma_func=lambda x: gamma_func(x, *gamma_opt),
                                 cycles=cycles, start_time=start_sim_time)
    if show_pictures:
        simulation_x = np.arange(training_range[1] + 1, training_range[1] + 1 + cycles)
        plt.plot(simulation_x, simulated['Infected'], label='Infected simulated')
        plt.plot(dataset.index, dataset['Infected'], label='Infected real')
        plt.title(f'Simulation start time={training_range[1] + 1}\n'
                  f'training range={training_range}\n'
                  f'simulated cycles={cycles}')
        plt.legend()
        plt.show()
    if return_result:
        return simulated


# %%

# ---------- Исполнение -------------------------------------------------------

source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'
frame = pd.read_csv(os.path.join(source_folder, file_name))
info_frame = pd.read_csv(os.path.join(source_folder, info_file_name))
region_frame = frame[frame['Region/City'] == REGION]
region_population = int(info_frame[info_frame['Region'] == REGION]['Population'])
region_frame = add_stats(region_frame, region_population)
if TO_SAVE:
    region_frame.to_excel(os.path.join(output_folder, f'{REGION}.xlsx'))
simulate_graphics(region_frame, region_population, (1, 90), cycles=15, gamma_func=func_lin)
# %%
simulate_graphics(region_frame, region_population, (1, 70), cycles=15,
                  beta_func=func_neg_exp, gamma_func=func_lin)

# %%
region_frame
# %%
import pandas as pd

s = pd.Series([0, 2, 5, 8, 13])
# %%
-s.diff(-2).shift(1)/2
# %%
plt.plot(region_frame.index, region_frame['dI'] / region_frame['Infected'])
plt.show()
# %%
