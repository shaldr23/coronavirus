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

REGION = 'Республика Татарстан'
TO_SAVE = False  # сохранять ли данные в excel

# ---------- Функции -------------------------------------------------

def add_stats(df: pd.DataFrame, population):
    """
    Добавление данных в DataFrame S, I, R, Beta, Gamma.
    Данные для каждого дня последовательности не требуются.
    Нужны лишь кумулятивные данные (заражений, смертей, выздоровлений) и даты.
    Если в какой-то день количество общих случаев не меняется -
    эта строчка игнорируется, и дельта t, соответственно, увеличивается.
    Первая строчка данных в итоге удаляется,
    т.к. показателей dI, dR, dS для нее нет.
    """
    df = df[['Date', 'Confirmed', 'Deaths', 'Recovered']]
    df['Removed'] = df['Deaths'] + df['Recovered']
    df['Infected'] = df['Confirmed'] - df['Removed']
    df['Suspected'] = population - df['Infected'] - df['Removed']
    dConfirmed = df['Confirmed'].diff()
    df = df[dConfirmed != 0]  # Убрали дни, где общее количество случаев не меняется
    dI = df['Infected'].diff()
    dR = df['Removed'].diff()
    dt = pd.to_datetime(df['Date']).diff().dt.days
    df['Gamma'] = dR / dt / df['Infected']
    df['Beta'] = (dI / dt + df['Infected'] * df['Gamma']) * population / (df['Infected'] * df['Suspected'])
    df['R_0'] = df['Beta'] / df['Gamma']
    df = df.iloc[1:]  # удаляется первая строчка с NaN в dI, dR, dS
    
    # обновляем индекс, учитывая пропущенные значения
    index = dt.iloc[1:].cumsum()
    index = index - index.iloc[0] + 1
    df.index = index
    df.index.name = 'index'
    return df


def simulate_dynamics(P: 'population', I: 'infected',
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


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def simulate_graphics(dataset: pd.DataFrame,
                      population,
                      training_range: '(start_index, end_index)',
                      cycles=100,
                      beta_func=func_neg_exp,
                      gamma_func=func_lin,
                      smoothing=True,
                      show_pictures=True,
                      return_result=False):
    """
    Функция, осуществляющая подбор уравнений для Beta и Gamma,
    использующая функцию simulate_dynamics для симуляции заражения с некоторого
    момента времени и строящая графики.
    """
    training_set = dataset.loc[training_range[0]:training_range[1] + 1]
    training_set.reset_index(inplace=True)  # сбрасываем индекс, т.к. эксп. регрессия капризная
    xdata = training_set.index

    y_beta = training_set['Beta']
    y_gamma = training_set['Gamma']
    if not smoothing:
        beta_opt = curve_fit(beta_func, xdata, y_beta)[0]
        gamma_opt = curve_fit(gamma_func, xdata, y_gamma)[0]
    else:
        y_beta_smoothed = smooth(y_beta, 5)
        y_gamma_smoothed = smooth(y_gamma, 5)
        beta_opt = curve_fit(beta_func, xdata, y_beta_smoothed)[0]
        gamma_opt = curve_fit(gamma_func, xdata, y_gamma_smoothed)[0]
    # Строим графики для Beta и Gamma, если нужно:
    if show_pictures:
        # График для Beta
        y_beta_fitted = beta_func(xdata, *beta_opt)
        r2 = r2_score(y_beta, y_beta_fitted)
        plt.plot(xdata, y_beta_fitted, 'r-',
                 label=f'params: {tuple(round(opt, 5) for opt in beta_opt)}')
        plt.plot(xdata, y_beta, label='real')
        if smoothing:
            plt.plot(xdata, y_beta_smoothed, label='smoothed')
        plt.legend()
        plt.title(f'Beta: {beta_func.__name__}, R2={r2}')
        plt.show()
        # График для Gamma
        y_gamma_fitted = gamma_func(xdata, *gamma_opt)
        r2 = r2_score(y_gamma, y_gamma_fitted)
        plt.plot(xdata, y_gamma_fitted, 'r-',
                 label=f'params: {tuple(round(opt, 5) for opt in gamma_opt)}')
        plt.plot(xdata, y_gamma, label='real')
        if smoothing:
            plt.plot(xdata, y_gamma_smoothed, label='smoothed')
        plt.legend()
        plt.title(f'Gamma: {gamma_func.__name__}, R2={r2}')
        plt.show()

    # Симуляция
    init_vals = dataset.loc[training_range[1]]  # From here we take I and R
    start_sim_time = training_set.index[-1] + 1
    simulated = simulate_dynamics(population, init_vals['Infected'], init_vals['Removed'],
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


def regional_simulation(frame, info_frame, region_name):
    """
    docstring
    """
    pass

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
simulate_graphics(region_frame, region_population, (85, 110), cycles=15,
                  beta_func=func_neg_exp, gamma_func=func_lin,
                  smoothing=True)

# %%
simulate_graphics(region_frame, region_population, (1, 75), cycles=15,
                  beta_func=func_neg_exp, gamma_func=func_lin,
                  smoothing=False)

# %%
