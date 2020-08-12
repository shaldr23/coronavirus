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

# ---------- Функции -------------------------------------------------

def add_stats(df: pd.DataFrame, population):
    """
    Добавление данных в DataFrame S, I, R, Beta, Gamma.
    Формулы работают, если есть данные для каждого дня в последовательности.
    Есть данные количества случаев в день (заражений, смертей, выздоровлений)
    Можно потом улучшить.
    """
    df.reset_index(inplace=True)
    df.index = df.index + 1 # Для нумерации с начала
    df['time_delta'] = pd.to_datetime(df['Date']).diff()
    assert all(df['time_delta'].dropna() == pd.Timedelta('1 days')) # удостоверимся, что разница между датами = 1 день
    df['Removed'] = df['Recovered'] + df['Deaths']
    df['Infected'] = df['Confirmed'] - df['Removed']
    df['Suspected'] = population - df['Infected'] - df['Removed']
    df['Gamma'] = (df['Day-Recovered'] + df['Day-Deaths']) / df['Infected']
    day_infected = df['Day-Confirmed'] - df['Day-Recovered'] - df['Day-Deaths']
    df['Beta'] = (day_infected + df['Infected'] * df['Gamma']) * population / (df['Infected'] * df['Suspected'])
    df['R_0'] = df['Beta'] / df['Gamma']


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


def func_exp(x, a, b):
    return a * np.exp(-b * x)


def func_lin(x, a, b):
    return a * x + b


def func_log(x, a, b, c, d):
    return a * np.log(b * x + c) + d


def func_logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def simulate_graphics(dataset: pd.DataFrame,
                      population,
                      training_range: '(start_index, end_index)',
                      cycles=100,
                      beta_func=func_exp,
                      gamma_func=func_lin,
                      show_pictures=True):
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
        plt.plot(dataset.index, moscow_frame['Infected'], label='Infected real')
        plt.title(f'Simulation start time={training_range[1] + 1}\n'
                  f'training range={training_range}\n'
                  f'simulated cycles={cycles}')
        plt.legend()
        plt.show()
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
add_stats(region_frame, region_population)
region_frame.to_excel(os.path.join(output_folder, f'{REGION}.xlsx'))

# %%
simulate_graphics(region_frame, region_population, (5, 40), cycles=30, gamma_func=func_lin)

# %%
simulate_graphics(region_frame, region_population, (5, 50), cycles=30, gamma_func=func_lin)
