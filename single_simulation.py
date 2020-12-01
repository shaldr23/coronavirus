# %%
"""
Симуляция одного временного отрезка
"""
import pandas as pd
import os
import tools
import importlib
importlib.reload(tools)
from tools import add_stats, simulate_graphics

# ---- Начальные параметры ----------------------

REGION = 'Москва'
TO_SAVE = False  # сохранять ли промежуточные данные в excel
SMOOTH_POINTS = 5
LAST_TRAINING_POINT = 20
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
simulate_graphics(region_frame, region_population, (1, LAST_TRAINING_POINT),
                  cycles=CYCLES, smooth_points=SMOOTH_POINTS)
