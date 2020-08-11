# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
# from helpack.timetools import time_it

source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'

source_file = os.path.join(source_folder, file_name)
frame = pd.read_csv(source_file)

# %%
def add_stats(df: pd.DataFrame, population):
    """
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


moscow_frame = frame[frame['Region/City'] == 'Москва']
moscow_population = 12692466
add_stats(moscow_frame, moscow_population)
output_file = os.path.join(output_folder, 'moscow.xlsx')
moscow_frame.to_excel(output_file)

# %%
# Симуляция с экспоненциальной beta и линейной gamma
# Подбор функции для beta


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


# функции для регрессии 
def func_exp(x, a, b):
    return a * np.exp(-b * x)


def func_lin(x, a, b):
    return a * x + b

# главная функция
def simulate_graphics(dataset: pd.DataFrame, population,
                      training_range: '(start_index, end_index)',
                      cycles=100,
                      show_pictures=True):
    """
    Description
    """
    training_set = dataset.loc[training_range[0]:training_range[1] + 1]
    old_index = training_set.index
    training_set.reset_index(inplace=True)  # сбрасываем индекс, т.к. эксп. регрессия капризная
    xdata = training_set.index

    y_beta = training_set['Beta']
    beta_opt = curve_fit(func_exp, xdata, y_beta)[0]
    y_gamma = training_set['Gamma']
    gamma_opt = curve_fit(func_lin, xdata, y_gamma)[0]

    # Строим графики для Beta и Gamma, если нужно:
    if show_pictures:
        # График для Beta
        y_beta_fitted = func_exp(xdata, *beta_opt)
        r2 = r2_score(y_beta, y_beta_fitted)
        plt.plot(xdata, y_beta_fitted, 'r-',
                label='fit: a=%5.3f, b=%5.3f' % tuple(beta_opt))
        plt.plot(xdata, y_beta, label='real')
        plt.legend()
        plt.title(f'Beta: экспонента, R2={r2}')
        plt.show()

        # График для Gamma
        y_gamma_fitted = func_lin(xdata, *gamma_opt)
        r2 = r2_score(y_gamma, y_gamma_fitted)
        plt.plot(xdata, y_gamma_fitted, 'r-',
                 label='fit: a=%5.3f, b=%5.3f' % tuple(gamma_opt))
        plt.plot(xdata, y_gamma, label='real')
        plt.legend()
        plt.title(f'Gamma: линейная, R2={r2}')
        plt.show()

    # Симуляция
    init_vals = dataset.loc[training_range[1]]  # From here we take I and R
    start_sim_time = training_set.index[-1] + 1
    simulated = simulate_dynamic(population, init_vals['Infected'], init_vals['Removed'],
                                 beta_func=lambda x: func_exp(x, *beta_opt),
                                 gamma_func=lambda x: func_lin(x, *gamma_opt),
                                 cycles=cycles, start_time=start_sim_time)
    simulation_x = np.arange(training_range[1] + 1, training_range[1] + 1 + cycles)
    plt.plot(simulation_x, simulated['Infected'], label='Infected simulated')
    plt.plot(dataset.index, moscow_frame['Infected'], label='Infected real')
    plt.title(f'Simulation start time={training_range[1] + 1}\n'
              f'training range={training_range}\n'
              f'simulated cycles={cycles}')
    plt.legend()
    plt.show()


# %%
simulate_graphics(moscow_frame, 12692466, (11, 40), cycles=50)

# %%
simulate_graphics(moscow_frame, 12692466, (20, 50), cycles=30)
