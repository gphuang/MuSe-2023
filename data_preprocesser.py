import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler

import config

def main():
    dir_feat = os.path.join(config.BASE_PATH, 'c3_muse_personalisation/feature_segments')
    dir_label = os.path.join(config.BASE_PATH, 'c3_muse_personalisation/label_segments')
    dir_biosignals = os.path.join(dir_feat, 'biosignals')
    dir_out = dir_biosignals + '_normalized'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for _biosignal in ['ECG','resp','BPM']:
        _dir = os.path.join(dir_feat, _biosignal) + '_normalized'
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    # todo check if 'save' option in load_data has saved similar files in results/data_muse/personalisation
    dir_list = os.listdir(dir_biosignals)
    # Standardizing data
    std_scaler = StandardScaler()
    for _csv in dir_list:
        df = pd.read_csv(os.path.join(dir_biosignals, _csv))
        df[['ECG','resp','BPM']] = std_scaler.fit_transform(df[['ECG','resp','BPM']])
        df.to_csv(os.path.join(dir_out, _csv), index=False)
        for _biosignal in ['ECG','resp','BPM']:
            _dir = os.path.join(dir_label, _biosignal) + '_normalized'
            _df = df[['timestamp','subject_id', _biosignal]]
            _df = _df.rename(columns={_biosignal: 'value'}) 
            _df.to_csv(os.path.join(_dir, _csv), index=False)
    
if __name__ == '__main__':
    main()
