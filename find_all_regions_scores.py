# %%
"""
Определение точности предсказаний для всех регионов
"""
import pandas as pd
import os
import tools
import importlib
importlib.reload(tools)
from tools import add_stats, multiple_simulate_graphics

# ---- Начальные параметры ----------------------

SMOOTH_POINTS = 5
FIRST_TRAINING_END = 20
TRAINING_END_INCREMENT = 1
CYCLES = 15
SAVE_FILE = 'scores.xlsx'
source_folder = 'data/source'
output_folder = 'data/output'
file_name = 'covid19-russia-cases-scrf.csv'
info_file_name = 'regions-info.csv'
file_path = os.path.join(output_folder, SAVE_FILE)

# ---- Исполнение ---------------------------------

frame = pd.read_csv(os.path.join(source_folder, file_name))
info_frame = pd.read_csv(os.path.join(source_folder, info_file_name))
scores_list = []
for num, region in enumerate(frame['Region/City'].unique(), start=1):
    region_frame = frame[frame['Region/City'] == region]
    region_population = int(info_frame[info_frame['Region'] == region]['Population'])
    region_frame = add_stats(region_frame, region_population)
    scores = multiple_simulate_graphics(region_frame, region_population,
                                        first_training_end=FIRST_TRAINING_END,
                                        training_end_increment=TRAINING_END_INCREMENT,
                                        return_result=True,
                                        smooth_points=SMOOTH_POINTS,
                                        show_pictures=False)
    scores['region'] = region
    scores_list.append(scores)
    if not num % 10:
        print(f'Обработано {num} регионов.')
scores_frame = pd.DataFrame(scores_list)
scores_frame.to_excel(file_path)
