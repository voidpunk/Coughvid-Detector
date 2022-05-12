import os
import shutil
import json
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from coughvid_functions import segment_cough, preprocess_cough, compute_segment_SNR
from helper_functions import plot_multiple_views

"""
Functions changed and generalized after data generation, may be bugged!
"""


def printer(target=10, base_path='../data/', plot=False):
    c = 0
    for file in os.listdir(base_path):
        if file.endswith(".json"):
            with open(base_path + file) as f:
                data = f.read()
            print(data)
            if plot:
                wav = base_path + file.replace('.json', '.wav')
                plot_multiple_views(wav)
            c += 1
        if c == target:
            break


def directoryzer(base_path='../data/', counter=600):
    c = counter
    n_cycle = 0
    os.mkdir(base_path + 'orginal')
    base_path = base_path + 'original/'
    directory = f'{n_cycle}-{c}'
    os.mkdir(base_path + directory)
    for file in tqdm(os.listdir(base_path)):
        if c > 0:
            if file.endswith(".json"):
                shutil.copy(base_path + file, base_path + directory + '/' + file)
                shutil.copy(base_path + file.replace('.json', '.wav'), base_path + directory + '/' + file.replace('.json', '.wav'))
            # elif file.endswith(".wav"):
            #     shutil.move(base_path + file, base_path + directory + '/' + file)
            #     shutil.move(base_path + file.replace('.wav', '.json'), base_path + directory + '/' + file.replace('.wav', '.json'))
            c -= 1
        else:
            c = counter
            n_cycle += 1
            directory = f'{n_cycle*c}-{n_cycle*c+c}'
            os.mkdir(base_path + directory)


def processor(target=-1, base_path='../data/original/', clean=False):
    for directory in tqdm(os.listdir(base_path)):
        data = {
            'uuid': [],
            'detection': [],
            'status': [],
            'age': [],
            'gender': [],
            'condition': [],
            'symptoms': [],
            'coordinates': [],
            'snr': [],
            'rate': [],
            'spectrum': [],
            'el1': [],
            'el2': [],
            'el3': [],
            'el4': []
        }
        c = 0
        path = base_path + directory + '/'
        for file in tqdm(os.listdir(path)):
            if file.endswith(".json"):
                uuid = file.replace('.json', '')
                with open(path + file) as f:
                    entry = json.load(f)
                wav_path = path + file.replace('.json', '.wav')
                wav, sr = librosa.load(wav_path)
                segments, mask = segment_cough(wav, sr)
                if len(segments) > 0:
                    for segment in segments:
                        prepro_segment, _ = preprocess_cough(segment, sr)
                        spectrum = librosa.stft(prepro_segment)
                        data['uuid'].append(uuid)
                        data['spectrum'].append(spectrum.tolist())
                        data['rate'].append(sr)
                        data['snr'].append(compute_segment_SNR(wav, sr, mask))
                        data['detection'].append(entry['cough_detected'])
                        data['status'].append(entry['status'] if 'status' in entry else 'NA')
                        data['age'].append(entry['age'] if 'age' in entry else 'NA')
                        data['gender'].append(entry['gender'] if 'gender' in entry else 'NA')
                        data['condition'].append(entry['respiratory_condition'] if 'respiratory_condition' in entry else 'NA')
                        data['symptoms'].append(entry['fever_muscle_pain'] if 'fever_muscle_pain' in entry else 'NA')
                        data['el1'].append(entry['el1'] if 'el1' in entry else 'NA')
                        data['el2'].append(entry['el2'] if 'el2' in entry else 'NA')
                        data['el3'].append(entry['el3'] if 'el3' in entry else 'NA')
                        data['el4'].append(entry['el4'] if 'el4' in entry else 'NA')
                        data['coordinates'].append((entry['latitude'], entry['longitude']) if 'latitude' in entry else 'NA')
                else:
                    # update data
                    data['uuid'].append(uuid)
                    data['spectrum'].append([])
                    data['rate'].append(sr)
                    data['snr'].append(compute_segment_SNR(wav, sr, mask))
                    data['detection'].append(entry['cough_detected'])
                    data['status'].append(entry['status'] if 'status' in entry else 'NA')
                    data['age'].append(entry['age'] if 'age' in entry else 'NA')
                    data['gender'].append(entry['gender'] if 'gender' in entry else 'NA')
                    data['condition'].append(entry['respiratory_condition'] if 'respiratory_condition' in entry else 'NA')
                    data['symptoms'].append(entry['fever_muscle_pain'] if 'fever_muscle_pain' in entry else 'NA')
                    data['el1'].append(entry['el1'] if 'el1' in entry else 'NA')
                    data['el2'].append(entry['el2'] if 'el2' in entry else 'NA')
                    data['el3'].append(entry['el3'] if 'el3' in entry else 'NA')
                    data['el4'].append(entry['el4'] if 'el4' in entry else 'NA')
                    data['coordinates'].append((entry['latitude'], entry['longitude']) if 'latitude' in entry else 'NA')
                c += 1
            if c == target:
                break
        data = pd.DataFrame(data).to_csv(path + 'data.csv')
    if clean:
        print('Cleaning...')
        for el in os.listdir(base_path):
            if not os.path.isdir(base_path + el):
                os.remove(base_path + el)


def collector(base_path='../data/', orig_path='../data/original/'):
    frag_dir = os.mkdir(base_path + 'fragmented')
    c = 1
    for directory in tqdm(os.listdir(orig_path)):
        shutil.copy(f'{frag_dir}/{directory}/data.csv', f'{frag_dir}/data{c}.csv')
        c += 1


if __name__ == '__main__':
    printer()
    directoryzer()
    processor()
    collector()