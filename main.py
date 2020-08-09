# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from helpack.timetools import time_it

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
    df['time_delta'] = pd.to_datetime(df['Date']).diff()
    assert all(df['time_delta'].dropna() == pd.Timedelta('1 days')) # удостоверимся, что разница между датами = 1 день
    df['Infected'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    df['Suspected'] = population - df['Infected'] - df['Deaths'] - df['Recovered']
    df['Gamma'] = df['Day-Recovered'] / df['Infected']
    df['Mu'] = df['Day-Deaths'] / df['Infected']
    df['Beta'] = (df['Day-Confirmed'] + df['Day-Recovered'] + df['Day-Deaths']) * population / (df['Infected'] * df['Suspected'])
    df['Rt'] = df['Beta'] / (df['Gamma'] + df['Mu'])


moscow_frame = frame[frame['Region/City'] == 'Москва']
moscow_population = 12692466
add_stats(moscow_frame, moscow_population)
output_file = os.path.join(output_folder, 'moscow.xlsx')
moscow_frame.to_excel(output_file)

# %%
# Калибровка параметров


def simulate(P: 'population', I: 'infected',
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


def pick_params(frame: 'pd.DataFrame real SIR data',
                P: 'population',
                beta_range: '[min, max, step]',
                gamma_range: '[min, max, step]',
                cycles_continue=0):
    I = frame['Infected'].iloc[0]
    R = frame['Removed'].iloc[0]
    cycles = len(frame) - 1
    min_error = float('inf')
    for beta in range(*beta_range):
        for gamma in range(*gamma_range):
            sim_data = simulate(P, I, R, beta, gamma, cycles)
            error = mse(frame['Infected'], sim_data['Infected'])
            if error < min_error:
                min_error = error
                best_beta = beta
                best_gamma = gamma
                best_sim_data = sim_data
    if cycles_continue:
        best_sim_data = simulate(P, I, R, best_beta, best_gamma,
                                 cycles + cycles_continue)
    result = {'error': min_error,
              'beta': best_beta,
              'gamma': best_gamma,
              'simulation_data': best_sim_data}
    return result

# %%
time_it(simulate, 10, 100000, 5, 0, 0.05, 0.01, cycles=500)


# %%
# simulation


t1 = time_it(simulate, 10, 100000, 5, 0, 0.05, 0.01, cycles=500)
t2 = time_it(simulate_2, 10, 100000, 5, 0, 0.05, 0.01, cycles=500)
print(t1, t2)
# %%
simulate_2(100000, 5, 0, 0.05, 0.01, cycles=500)

# %%
predictions

# %%
predictions.to_excel(os.path.join(output_folder, 'predictions.xlsx'))

# %%
mse([1,2,3], [1.5, 3, 4])

# %%
