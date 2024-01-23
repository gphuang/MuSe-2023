import os, sys, re
import pandas as pd

# from config import LOG_FOLDER, TASKS
OUTPUT_PATH = os.path.join('/scratch/work/huangg5/muse/MuSe-2023', 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
TASKS = ['mimic', 'humor', 'personalisation']
print(LOG_FOLDER)
print(TASKS)

_data = []
for task in TASKS:
    directory = os.path.join(LOG_FOLDER, task) #os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            feature = filename.replace('[','').replace(']','').split('_')[2]
            if task == 'personalisation':
                output = filename.replace('[','').replace(']','').split('_')[3]
            _inf = os.path.join(directory, filename)
            with open(_inf, 'rb') as input:
                _text = input.read().decode('utf-8')
                # ID/Seed 105 | Best [Val Mean Pearsons]: 0.0691 |
                regex = re.compile(r'Seed\s(\d+)\s\|\sBest\s\[(.+)\]\:\s(.+)\s\|')
                _df = pd.DataFrame(regex.findall(_text), columns=['Seed', 'Metric', 'Value']) 
            _df['feature'] = feature 
            _df['task'] = task
            if task == 'personalisation':
                _df['output'] = output
            _data.append(_df)
            continue
        else:
            continue
df = pd.concat(_data)
print(df.shape)
print(df.head(3))
df.to_csv(f'./logs/val_performance.csv', index=False)
df = pd.read_csv(f'./logs/val_performance.csv')
print(df.shape)
print(df.head(3))
