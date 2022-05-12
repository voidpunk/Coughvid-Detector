import os
import shutil
import json
import librosa
from PIL import Image
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


def printer(base_path, target=10, plot=False):
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


def directoryzer(base_path, new_path=None, counter=600, clean=False):
    c = counter
    n_cycle = 0
    if new_path is not None:
        target_path = os.path.join(base_path, new_path)
        os.mkdir(target_path)
    else:
        target_path = base_path
    directory = f'{n_cycle}-{c}'
    os.mkdir(os.path.join(target_path, directory))
    for file in tqdm(os.listdir(base_path)):
        if c > 0:
            if file.endswith(".json"):
                shutil.copy(
                    os.path.join(base_path, file),
                    os.path.join(target_path, directory, file)
                )
                shutil.copy(
                    os.path.join(base_path, file.replace('.json', '.wav')),
                    os.path.join(target_path, directory, file.replace('.json', '.wav'))
                )
            c -= 1
        else:
            c = counter
            n_cycle += 1
            directory = f'{n_cycle*c}-{n_cycle*c+c}'
            os.mkdir(os.path.join(target_path, directory))
    if clean:
        print('Cleaning...')
        for el in tqdm(os.listdir(base_path)):
            if not os.path.isdir(os.path.join(base_path, el)):
                os.remove(os.path.join(base_path, el))



def processor(target=-1, base_path='../data/divided/', clean=False):
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
                        data['el1'].append(entry['expert_labels_1'] if 'expert_labels_1' in entry else 'NA')
                        data['el2'].append(entry['expert_labels_2'] if 'expert_labels_2' in entry else 'NA')
                        data['el3'].append(entry['expert_labels_3'] if 'expert_labels_3' in entry else 'NA')
                        data['el4'].append(entry['expert_labels_4'] if 'expert_labels_4' in entry else 'NA')
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
                    data['el1'].append(entry['expert_labels_1'] if 'expert_labels_1' in entry else 'NA')
                    data['el2'].append(entry['expert_labels_2'] if 'expert_labels_2' in entry else 'NA')
                    data['el3'].append(entry['expert_labels_3'] if 'expert_labels_3' in entry else 'NA')
                    data['el4'].append(entry['expert_labels_4'] if 'expert_labels_4' in entry else 'NA')
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


def collector(base_path='../data/', orig_path='../data/divided/'):
    frag_dir = os.mkdir(base_path + 'fragmented')
    c = 1
    for directory in tqdm(os.listdir(orig_path)):
        shutil.copy(f'{frag_dir}/{directory}/data.csv', f'{frag_dir}/data{c}.csv')
        c += 1


def generator(base_path='../data/extracted/', target=5, plot=False):
    c = 0
    for file in tqdm(os.listdir(base_path)):
        if c == target:
            break
        if file.endswith(".wav"):
            wav, sr = librosa.load(base_path + file)
            segments, _ = segment_cough(wav, sr)
            if len(segments) > 0:
                    for segment in segments:
                        prepro_segment, _ = preprocess_cough(segment, sr)
                        spectrum = librosa.stft(prepro_segment)
                        print(type(spectrum), spectrum.shape)
                        if plot:
                            _, ax = plt.subplots(figsize=(10,6))
                            librosa.display.specshow(spectrum, y_axis='log', x_axis='s', sr=sr, ax=ax)
                            plt.show()
        c += 1


# printer('../data/original/')
directoryzer('../data/public_dataset', clean=True)
# processor()
# collector()
# generator()
