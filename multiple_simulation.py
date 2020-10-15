# %%
"""
Симуляция множественных временных отрезков
"""
import pandas as pd
import os
import tools
import importlib
importlib.reload(tools)
from tools import add_stats, multiple_simulate_graphics

# ---- Начальные параметры ----------------------

REGION = 'Москва'
TO_SAVE = False  # сохранять ли промежуточные данные в excel
SMOOTH_POINTS = 5
FIRST_TRAINING_END = 20
TRAINING_END_INCREMENT = 1
CYCLES = 15
source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'

# ---- Исполнение ---------------------------------

frame = pd.read_csv(os.path.join(source_folder, file_name))
info_frame = pd.read_csv(os.path.join(source_folder, info_file_name))
region_frame = frame[frame['Region/City'] == REGION]
region_population = int(info_frame[info_frame['Region'] == REGION]['Population'])
region_frame = add_stats(region_frame, region_population)
if TO_SAVE:
    region_frame.to_excel(os.path.join(output_folder, f'{REGION}.xlsx'))

multiple_simulate_graphics(region_frame, region_population,
                           first_training_end=FIRST_TRAINING_END,
                           training_end_increment=TRAINING_END_INCREMENT,
                           return_result=True,
                           smooth_points=SMOOTH_POINTS)

# %%
