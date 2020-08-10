# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

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
    Return dict, not DataFrame, because it is MUCH faster
    """
    S = P - I - R
    data = {'Suspected': [S],
            'Infected': [I],
            'Removed': [R]}
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
                     start_time=0, cycles=100) -> dict:
    """
    Function to simulate SIR model. beta and gamma depend on time.
    Return dict, not DataFrame, because it is MUCH faster
    Подумать, чтобы смещения по времени не было.
    """
    S = P - I - R
    data = {'Suspected': [S],
            'Infected': [I],
            'Removed': [R]}
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
                best_r2 = r2 # на деле это r^2, соответствующий лучшему mse
    if cycles_continue:
        best_sim_data = simulate_static(P, I, R, best_beta, best_gamma,
                                        cycles + cycles_continue)
    result = {'error': min_error,
              'r2': best_r2,
              'beta': best_beta,
              'gamma': best_gamma,
              'simulation_data': best_sim_data}
    return result

# %%
# simulation with custom parameters
var = moscow_frame.iloc[0] # берем начальные значения из Московсикх данных
simulated = simulate_static(12692466, var['Infected'], var['Removed'],
                            beta=.07, gamma=.01, cycles=600)
plt.plot(simulated['Infected'], label='Infected')
plt.plot(simulated['Suspected'], label='Suspected')
plt.plot(simulated['Removed'], label='Removed')
plt.legend()
plt.show()
# %%
FIT_POINTS = 30 # количество первых точек, по которым производить подгонку параметров
params = pick_params(moscow_frame.iloc[:FIT_POINTS], 12692466, [0, 1, 0.01], [0, .1, 0.01],
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


# %%
params['beta']

# %%
var = moscow_frame.iloc[0]
cycles = len(moscow_frame) - 1
simulated = simulate_static(12692466, var['Infected'], var['Removed'],
                            beta=.07, gamma=.001, cycles=cycles)
plt.plot(moscow_frame['Infected'], label='real_data')
plt.plot(simulated['Infected'], label='simulation')
plt.legend()
plt.show()
# %%
plt.plot(simulated['Infected'], label='simulation')

# %%
params['beta']

# %%
from scipy.optimize import curve_fit

frame = moscow_frame.iloc[40:]
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func2(x, a, b):
    return a * x + b

def func3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
xdata = np.array(frame.index)
ydata = frame['Beta']

popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.plot(xdata, ydata, label='real')
plt.legend()
plt.show()

popt, pcov = curve_fit(func2, xdata, ydata)
plt.plot(xdata, func2(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.plot(xdata, ydata, label='real')
plt.legend()
plt.show()

popt, pcov = curve_fit(func3, xdata, ydata)
plt.plot(xdata, func3(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
plt.plot(xdata, ydata, label='real')
plt.legend()
plt.show()
# %%
np.array(moscow_frame.index)

# %%
simulated = simulate_dynamic(12692466, var['Infected'], var['Removed'],
                             beta_func=lambda x: -0.0015*x + 0.213,
                             gamma_func=lambda x: 0.00015*x + 0.0086,
                             cycles=cycles)


# %%
plt.plot(simulated['Infected'])
plt.plot(moscow_frame['Infected'])

# %%
