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
# Калибровка параметров


def simulate_static(P: 'population', I: 'infected',
                    R: 'removed', beta, gamma,
                    cycles=100) -> dict:
    """
    Function to simulate SIR model. beta and gamma are static.
    Return dict, not DataFrame, because it is MUCH faster.
    Начальные данные не включаются в симуляцию
    """
    S = P - I - R
    data = {'Suspected': [],
            'Infected': [],
            'Removed': []}
    for i in range(cycles):
        S = S - beta * I * S / P
        I = I + beta * I * S / P - gamma * I
        R = R + gamma * I
        data['Suspected'].append(S)
        data['Infected'].append(I)
        data['Removed'].append(R)
    return data


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

def pick_params(frame: 'pd.DataFrame real SIR data',
                P: 'population',
                beta_range: '[min, max, step]',
                gamma_range: '[min, max, step]',
                cycles_continue=0) -> dict:
    """
    Функция для подбора статичных параметров beta и gamma
    по наименьшей среднеквадратичной ошибке.
    """
    I = frame['Infected'].iloc[0]
    R = frame['Removed'].iloc[0]
    cycles = len(frame) - 1
    min_error = float('inf')
    for beta in np.arange(*beta_range):
        for gamma in np.arange(*gamma_range):
            sim_data = simulate_static(P, I, R, beta, gamma, cycles)
            error = mse(frame['Infected'], sim_data['Infected'])
            r2 = r2_score(frame['Infected'], sim_data['Infected'])
            if error < min_error:
                min_error = error
                best_beta = beta
                best_gamma = gamma
                best_sim_data = sim_data
                best_r2 = r2  # на деле это r^2, соответствующий лучшему mse
    if cycles_continue:
        best_sim_data = simulate_static(P, I, R, best_beta, best_gamma,
                                        cycles + cycles_continue)
    result = {'error': min_error,
              'r2': best_r2,
              'beta': best_beta,
              'gamma': best_gamma,
              'simulation_data': best_sim_data}
    return result


def pick_params_sum_r2(frame: 'pd.DataFrame real SIR data',
                       P: 'population',
                       beta_range: '[min, max, step]',
                       gamma_range: '[min, max, step]',
                       cycles_continue=0) -> dict:
    """
    Функция для подбора статичных параметров beta и gamma
    по максимальной сумме r2 infected и removed.
    """
    I = frame['Infected'].iloc[0]
    R = frame['Removed'].iloc[0]
    cycles = len(frame) - 1
    max_sum_r2 = 0
    for beta in np.arange(*beta_range):
        for gamma in np.arange(*gamma_range):
            sim_data = simulate_static(P, I, R, beta, gamma, cycles)
            r2_infected = r2_score(frame['Infected'], sim_data['Infected'])
            r2_removed = r2_score(frame['Removed'], sim_data['Removed'])
            sum_r2 = r2_infected + r2_removed
            if sum_r2 > max_sum_r2:
                max_sum_r2 = sum_r2
                best_beta = beta
                best_gamma = gamma
                best_sim_data = sim_data
    if cycles_continue:
        best_sim_data = simulate_static(P, I, R, best_beta, best_gamma,
                                        cycles + cycles_continue)
    result = {'sum_r2': max_sum_r2,
              'beta': best_beta,
              'gamma': best_gamma,
              'simulation_data': best_sim_data}
    return result

# %%
# simulation with custom parameters
var = moscow_frame.iloc[0]  # берем начальные значения из Московсикх данных
simulated = simulate_static(12692466, var['Infected'], var['Removed'],
                            beta=.07, gamma=.01, cycles=600)
plt.plot(simulated['Infected'], label='Infected')
plt.plot(simulated['Suspected'], label='Suspected')
plt.plot(simulated['Removed'], label='Removed')
plt.legend()
plt.show()
# %%
# Подгонка параметров по mse Infected
FIT_POINTS = 30  # количество первых точек, по которым производить подгонку параметров
params = pick_params(moscow_frame.iloc[:FIT_POINTS], 12692466, [0, 1, 0.01], [0, 1, 0.01],
                     cycles_continue=30)
beta = params['beta']
gamma = params['gamma']
mse_error = params['error']
r2 = params['r2']

plt.plot(moscow_frame['Infected'], label='real_data')
plt.plot(params['simulation_data']['Infected'],
         label=f'simulation beta={beta:.2f}, gamma={gamma:.2f}, mse={mse_error:.2f}, r2={r2:.2f}')
plt.legend()
plt.title(f'simulation with static gamma and beta FIT_POINTS={FIT_POINTS}')
plt.show()
# %%
# Подгонка параметров по максимальному суммарному r^2
FIT_POINTS = 30  # количество первых точек, по которым производить подгонку параметров
params = pick_params_sum_r2(moscow_frame.iloc[:FIT_POINTS], 12692466, [0, 1, 0.01], [0, 1, 0.01],
                            cycles_continue=30)
beta = params['beta']
gamma = params['gamma']
sum_r2 = params['sum_r2']

plt.plot(moscow_frame['Infected'], label='real_data')
plt.plot(params['simulation_data']['Infected'],
         label=f'simulation beta={beta:.2f}, gamma={gamma:.2f}, sum_r2={sum_r2:.2f}')
plt.legend()
plt.title(f'simulation with static gamma and beta FIT_POINTS={FIT_POINTS}')
plt.show()

# %%
# Симуляция с динамическими параметрами beta и gamma.
# Симуляция начинается с некоторого момента времени.

time = 0
var = moscow_frame.iloc[time]
simulated = simulate_dynamic(12692466, var['Infected'], var['Removed'],
                             beta_func=lambda x: -0.0018*x + 0.213,
                             gamma_func=lambda x: 0.0001*x + 0.0086,
                             cycles=300)
plt.plot(simulated['Infected'], label='Infected (simulated)')
plt.plot(moscow_frame['Infected'], label='Infected (real)')
plt.legend()
plt.show()

# %%
# Симуляция не с начала. Уже лучше.
time = 20  # время начальных данных, симуляция начинается с time + 1
cycles = 50
var = moscow_frame.loc[time]
simulated = simulate_dynamic(12692466, var['Infected'], var['Removed'],
                             beta_func=lambda x: -0.0018*x + 0.213,
                             gamma_func=lambda x: 0.0001*x + 0.0086,
                             cycles=cycles, start_time=time + 1)
plt.plot(np.arange(time + 1, time + 1 + cycles), simulated['Infected'])
plt.plot(moscow_frame.loc[time:].index, np.array(moscow_frame.loc[time:]['Infected']))
plt.title(f'start time={time}')
plt.show()


# %%
# Симуляция с экспоненциальной beta и линейной gamma
# Подбор функции для beta

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
simulate_graphics(moscow_frame, 12692466, (11, 100), cycles=30)

# %%
# Смотрим, какая зависимость у параметров Beta и Gamma
# Важно: удалить первые строчки из набора данных,
# иначе Beta не подгоняется.

frame = moscow_frame.iloc[10:]
frame.reset_index(inplace=True)
xdata = np.array(frame.index)
ydata = frame['Beta']


def func_exp(x, a, b):
    return a * np.exp(-b * x)

popt, pcov = curve_fit(func_exp, xdata, ydata)
y_fitted = func_exp(xdata, *popt)
r2 = r2_score(ydata, y_fitted)
plt.plot(xdata, y_fitted, 'r-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.plot(xdata, ydata, label='real')
plt.legend()
plt.title(f'Beta: экспонента, R2={r2}')
plt.show()




# %%
# Повторяем эксперимент в статье Бердутина
