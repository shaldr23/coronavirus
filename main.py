# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# ---------- Начальные параметры -------------------------------------

REGION = 'Москва'
TO_SAVE = True  # сохранять ли данные в excel
SMOOTH_POINTS = 10

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
              use_gradual_change=True):
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
    df = df[df['Confirmed'] != 0]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    if fill_values:
        full_dates = pd.date_range(df['Date'].min(), df['Date'].max())
        full_dates = pd.DataFrame({'Date': full_dates})
        full_dates['Date'] = full_dates['Date'].dt.date
        df = full_dates.merge(df, how='left', on='Date')
        df.fillna(method='ffill', inplace=True)
    if use_gradual_change:
        for col in ('Confirmed', 'Deaths', 'Recovered'):
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


def func_lin(x, a, b):
    return a * x + b


def smooth(y, box_pts):
    """
    Функция для сглаживания графиков
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def mape_score(y_true, y_pred):
    """
    mean_absolute_percentage_error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def simulate_graphics(dataset: pd.DataFrame,
                      population,
                      training_range: '(start_index, end_index)',
                      cycles=100,
                      beta_func=func_neg_exp,
                      gamma_func=func_lin,
                      correct_beta_coef=True,
                      smooth_points=5,
                      show_pictures=True,
                      return_result=False):
    """
    Функция, осуществляющая подбор уравнений для Beta и Gamma,
    использующая функцию simulate_dynamics для симуляции заражения с некоторого
    момента времени и строящая графики.
    correct_beta_coef: коррекция показателя степени в функции для beta,
    если beta растет: из правой половины отщепляем максимальные значения,
    пока beta не будет уменьшаться.
    """
    training_set = dataset.loc[training_range[0]:training_range[1] + 1]
    training_set.reset_index(inplace=True)  # сбрасываем индекс, т.к. эксп. регрессия капризная
    xdata = training_set.index

    y_beta = training_set['Beta']
    y_gamma = training_set['Gamma']
    if not smooth_points:
        beta_opt = curve_fit(beta_func, xdata, y_beta)[0]
        gamma_opt = curve_fit(gamma_func, xdata, y_gamma)[0]
    else:
        y_beta_smoothed = smooth(y_beta, smooth_points)
        y_gamma_smoothed = smooth(y_gamma, smooth_points)
        beta_opt = curve_fit(beta_func, xdata, y_beta_smoothed)[0]
        gamma_opt = curve_fit(gamma_func, xdata, y_gamma_smoothed)[0]
    if correct_beta_coef:
        beta_frame = pd.DataFrame({'xdata': xdata,
                                   'ydata': y_beta})
        while beta_opt[1] < 0:
            left_border = int((len(beta_frame) + 1)/2)
            idxmax = beta_frame.iloc[left_border:]['ydata'].idxmax()
            beta_frame = beta_frame.drop(idxmax)
            if not smooth_points:
                beta_opt = curve_fit(beta_func, beta_frame['xdata'], beta_frame['ydata'])[0]
            else:
                y_beta_corr_smoothed = smooth(beta_frame['ydata'], smooth_points)
                beta_opt = curve_fit(beta_func, beta_frame['xdata'], y_beta_corr_smoothed)[0]
    # Строим графики для Beta и Gamma, если нужно:
    if show_pictures:
        # График для Beta
        y_beta_fitted = beta_func(xdata, *beta_opt)
        r2 = r2_score(y_beta, y_beta_fitted)
        plt.plot(xdata, y_beta, '-o', label='Реальные данные')
        if smooth_points:
            plt.plot(xdata, y_beta_smoothed, label='Сглаженные данные')
            r2 = r2_score(y_beta_smoothed, y_beta_fitted)
        plt.plot(xdata, y_beta_fitted, 'r-',
                 label=f'Регрессионная кривая.\nПараметры: {tuple(round(opt, 5) for opt in beta_opt)}\n'
                 f'R2 = {r2:.3f}')
        plt.legend()
        plt.title('Построение графика для параметра \u03B2')
        plt.xlabel('Дни')
        plt.ylabel('Значение показателя \u03B2')
        plt.show()
        # График для Gamma
        y_gamma_fitted = gamma_func(xdata, *gamma_opt)
        r2 = r2_score(y_gamma, y_gamma_fitted)
        plt.plot(xdata, y_gamma, '-o', label='Реальные данные')
        if smooth_points:
            plt.plot(xdata, y_gamma_smoothed, label='Сглаженные данные')
            r2 = r2_score(y_gamma_smoothed, y_gamma_fitted)
        plt.plot(xdata, y_gamma_fitted, 'r-',
                 label=f'Регрессионная кривая.\nПараметры: {tuple(round(opt, 5) for opt in gamma_opt)}\n'
                 f'R2 = {r2:.3f}')
        plt.legend()
        plt.title('Построение графика для параметра \u03B3')
        plt.xlabel('Дни')
        plt.ylabel('Значение показателя \u03B3')
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
        plt.plot(dataset.index, dataset['Infected'], label='Реальные данные')
        r2 = r2_score(dataset['Infected'][simulation_x], simulated['Infected'])
        mape = mape_score(dataset['Infected'][simulation_x], simulated['Infected'])
        plt.plot(simulation_x, simulated['Infected'], 'r-',
                 label=f'Симуляция\nR2 = {r2:.3f}\nMAPE = {mape:.2f}')
        plt.title(f'Симуляция с дня {training_range[1] + 1}')
        plt.xlabel('Дни')
        plt.ylabel('Количество инфицированных')
        plt.legend()
        plt.show()
    if return_result:
        return simulated


def multiple_simulate_graphics(dataset: pd.DataFrame,
                               population,
                               first_training_end=30,
                               training_end_increment=15,
                               cycles=15,
                               beta_func=func_neg_exp,
                               gamma_func=func_lin,
                               smooth_points=5,
                               show_pictures=True,
                               return_result=True,
                               restrict_y=True):
    """
    !!! Реализовать вывод результата, м.б. одной циферкой.
    """
    all_sim_data = {'x': [], 'y': []}
    for training_end in range(first_training_end,
                              len(dataset) - cycles + 1,
                              training_end_increment):
        simulated = simulate_graphics(dataset,
                                      population,
                                      (1, training_end),
                                      cycles=cycles,
                                      beta_func=beta_func,
                                      gamma_func=gamma_func,
                                      smooth_points=smooth_points,
                                      show_pictures=False,
                                      return_result=True)
        all_sim_data['y'].append(simulated['Infected'])
        simulation_x = np.arange(training_end + 1, training_end + 1 + cycles)
        all_sim_data['x'].append(simulation_x)
        if training_end == first_training_end:
            label = 'Симуляция'
        else:
            label = None
        plt.plot(simulation_x, simulated['Infected'], 'r-',
                 label=label)
    plt.plot(dataset.index, dataset['Infected'], label='Реальные данные')
    plt.legend()
    if restrict_y:
        if type(restrict_y) in (int, float):
            max_y = restrict_y
        else:
            max_y = dataset['Infected'].max() * 2
        plt.ylim(top=max_y)
    plt.xlabel('Дни')
    plt.ylabel('Количество инфицированных')
    plt.title(f'Множественная симуляция с дня {first_training_end + 1}')
    plt.show()
    if return_result:
        return all_sim_data


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
simulate_graphics(region_frame, region_population, (1, 90), cycles=15,
                  smooth_points=5)
# %%
simulate_graphics(region_frame, region_population, (1, 20), cycles=15,
                  smooth_points=5)

# %%
multiple_simulate_graphics(region_frame, region_population,
                           first_training_end=20,
                           training_end_increment=1,
                           return_result=False,
                           restrict_y=False)

# %%
