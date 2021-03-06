import os
import json
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ast import literal_eval
import matplotlib.pyplot as plt
from operator import itemgetter
from modules.helper_functions import plot_multiple_views
from modules.coughvid_functions import (
    segment_cough,
    preprocess_cough,
    compute_segment_SNR
)


def printer(base_path, target=10, plot=False):
    c = 0
    for file in os.listdir(base_path):
        if file.endswith(".json"):
            with open(os.path.join(base_path, file)) as f:
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



def processor(base_path, target=-1, clean=False):
    for directory in tqdm(sorted(os.listdir(base_path))):
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
        path = os.path.join(base_path, directory)
        for file in tqdm(os.listdir(path)):
            if file.endswith(".json"):
                uuid = file.replace('.json', '')
                with open(os.path.join(path, file)) as f:
                    entry = json.load(f)
                wav_path = os.path.join(path, file.replace('.json', '.wav'))
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
        data = pd.DataFrame(data).to_csv(os.path.join(path, 'data.csv'))
    if clean:
        print('Cleaning...')
        for directory in tqdm(os.listdir(base_path)):
            for file in os.listdir(os.path.join(base_path, directory)):
                if not file.endswith('.csv'):
                    os.remove(os.path.join(base_path, directory, file))


def collector(base_path, dest_path=None, transfer=False):
    if dest_path is None:
        dest_path = os.path.join(base_path, 'fragments')
        os.mkdir(dest_path)
    i = 1
    for directory in tqdm(sorted(os.listdir(base_path))):
        if directory == 'fragments':
            continue
        if not transfer:
            shutil.copy(
                os.path.join(base_path, directory, 'data.csv'),
                os.path.join(dest_path, f'data{i}.csv')
            )
            i += 1
        else:
            shutil.move(
                os.path.join(base_path, directory, 'data.csv'),
                os.path.join(dest_path, f'data{i}.csv')
            )
            i += 1


def csver(base_path, dest_path=None, clean=False):
    if dest_path is None:
        dest_path = os.path.join(base_path, 'extracted')
    os.mkdir(dest_path)
    for file in tqdm(sorted(os.listdir(base_path))):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(base_path, file), index_col=0)
            df = df[
                (df.detection >= 0.8) & 
                (
                    (df.status.notna()) | 
                    (df.el1.notna())    | 
                    (df.el2.notna())    | 
                    (df.el3.notna())    | 
                    (df.el4.notna())
                )
            ]
            df.reset_index(drop=True, inplace=True)
            df.drop(
                columns=[
                    'uuid',
                    'detection',
                    'age',
                    'gender',
                    'condition',
                    'symptoms',
                    'coordinates',
                    'snr',
                    'rate',
                ],
                inplace=True
            )
            df.el1.fillna('na', inplace=True)
            df.el2.fillna('na', inplace=True)
            df.el3.fillna('na', inplace=True)
            df.el4.fillna('na', inplace=True)
            df.status.fillna('na', inplace=True)
            def get_diagnosis(el):
                if el == 'na':
                    return '0'
                elif el.find('upper_infection') != -1:
                    return '1'
                elif el.find('lower_infection') != -1:
                    return '2'
                elif el.find('COVID-19') != -1:
                    return '3'
            def get_Status(el):
                if el == 'na':
                    return '0'
                elif el.find('healthy') != -1:
                    return '1'
                elif el.find('symptomatic') != -1:
                    return '2'
                elif el.find('COVID-19') != -1:
                    return '3'
            df.el1 = df.el1.map(get_diagnosis)
            df.el2 = df.el2.map(get_diagnosis)
            df.el3 = df.el3.map(get_diagnosis)
            df.el4 = df.el4.map(get_diagnosis)
            df.status = df.status.map(get_Status)
            df.to_csv(os.path.join(dest_path, file))
    if clean:
        print('Cleaning...')
        for el in os.listdir(base_path):
            if not os.path.isdir(os.path.join(base_path, el)):
                os.remove(os.join.path(base_path, el))


def get_diagnosis(el):
    if str(el) == 'na':
        return '0'
    elif str(el).find('diagnosis') == -1:
        return '0'
    elif str(el).find('upper_infection') != -1:
        return '1'
    elif str(el).find('lower_infection') != -1:
        return '2'
    elif str(el).find('COVID-19') != -1:
        return '3'
    elif str(el).find('obstructive_disease') != -1:
        return '4'
    elif str(el).find('healthy_cough') != -1:
        return '5'
    else:
        print(el)
        raise Exception('Unknown diagnosis')


def get_Status(el):
    if el == 'na':
        return '0'
    elif el.find('healthy') != -1:
        return '1'
    elif el.find('symptomatic') != -1:
        return '2'
    elif el.find('COVID-19') != -1:
        return '3'
    else:
        print(el)
        raise Exception('Unknown status')


def get_code(meta):
    status = meta['status'] if 'status' in meta else 'na'
    el1 = meta['expert_labels_1'] if 'expert_labels_1' in meta else 'na'
    el2 = meta['expert_labels_2'] if 'expert_labels_2' in meta else 'na'
    el3 = meta['expert_labels_3'] if 'expert_labels_3' in meta else 'na'
    el4 = meta['expert_labels_4'] if 'expert_labels_4' in meta else 'na'
    if status == 'na' and el1 == 'na' and el2 == 'na' and el3 == 'na' and el4 == 'na':
        return 0
    status = get_Status(status)
    el1 = get_diagnosis(el1)
    el2 = get_diagnosis(el2)
    el3 = get_diagnosis(el3)
    el4 = get_diagnosis(el4)
    code = status + el1 + el2 + el3 + el4
    return code


def data_generator(base_path, dest_path=None, clean=False, target=-1, db=False, plot=False):
    c = 0
    # create output directory if needed
    if dest_path is None:
        dest_path = os.path.join(base_path, 'output')
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    # check if output directory is empty
    if len(os.listdir(dest_path)) != 0:
        raise Exception('Output folder is not empty!')
    for file in tqdm(os.listdir(base_path)):
        if file.endswith(".json"):
            # get filename
            filename = file[:-5]
            # read metadata
            with open(os.path.join(base_path, file)) as f:
                meta = json.load(f)
            # filter for cough detection
            if float(meta['cough_detected']) < 0.8:
                if clean:
                    # remove json and wav files
                    os.remove(os.path.join(base_path, f'{filename}.json'))
                    os.remove(os.path.join(base_path, f'{filename}.wav'))
                continue
            code = get_code(meta)
            # filter for status-diagnosis availability
            if code == 0:
                if clean:
                    # remove json and wav files
                    os.remove(os.path.join(base_path, f'{filename}.json'))
                    os.remove(os.path.join(base_path, f'{filename}.wav'))
                continue
            # read audio
            wav_path = os.path.join(base_path, file.replace('.json', '.wav'))
            wav, sr = librosa.load(wav_path)
            # process audio
            segments, _ = segment_cough(wav, sr)
            if len(segments) > 0:
                for segment in segments:
                    prepro_segment, _ = preprocess_cough(segment, sr)
                    spectrum = librosa.stft(prepro_segment)
                    if db:
                        spectrum = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
                    if plot:
                        fig, ax = plt.subplots(figsize=(10,6))
                        image = librosa.display.specshow(spectrum, y_axis='linear', x_axis='s', sr=sr, ax=ax)
                        plt.show()
                    # convert spectrum to image
                    spectrum = spectrum.astype(np.uint8)
                    img = Image.fromarray(spectrum)
                    img.save(os.path.join(dest_path, f'{c}-{code}.png'))
                    c += 1
            if clean:
                # remove json and wav files
                os.remove(os.path.join(base_path, f'{filename}.json'))
                os.remove(os.path.join(base_path, f'{filename}.wav'))
        if c == target:
            break


def clear_ambiguous(path):
    files = os.listdir(path)
    univoque = [
        el for el in files if
            el[-8:-4].count('1') == 1 and (
                el[-8:-4].count('2') == 0 and
                el[-8:-4].count('3') == 0 and
                el[-8:-4].count('4') == 0 and
                el[-8:-4].count('5') == 0
            ) or
            el[-8:-4].count('2') == 1 and (
                el[-8:-4].count('1') == 0 and
                el[-8:-4].count('3') == 0 and
                el[-8:-4].count('4') == 0 and
                el[-8:-4].count('5') == 0
            ) or
            el[-8:-4].count('3') == 1 and (
                el[-8:-4].count('1') == 0 and
                el[-8:-4].count('2') == 0 and
                el[-8:-4].count('4') == 0 and
                el[-8:-4].count('5') == 0
            ) or
            el[-8:-4].count('4') == 1 and (
                el[-8:-4].count('1') == 0 and
                el[-8:-4].count('2') == 0 and
                el[-8:-4].count('3') == 0 and
                el[-8:-4].count('5') == 0
            ) or
            el[-8:-4].count('5') == 1 and (
                el[-8:-4].count('1') == 0 and
                el[-8:-4].count('2') == 0 and
                el[-8:-4].count('3') == 0 and
                el[-8:-4].count('4') == 0
            ) or
            el[-8:-4].count('0') == 4 or
            el[-8:-4].count('1') == 4 or
            el[-8:-4].count('2') == 4 or
            el[-8:-4].count('3') == 4 or
            el[-8:-4].count('4') == 4 or
            el[-8:-4].count('5') == 4
    ]
    univoque_cut = [el[-8:-4] for el in univoque]
    for el in os.listdir(path):
        if el[-8:-4] not in univoque_cut:
            os.remove(os.path.join(path, el))


def label_del_func(file):
    # print(file)
    status = file.split('-')[1][0]
    code = file[-8:-4]
    if code.count('0') == 4:
        syn = 0
    else:
        count = (
            (1, code.count('1')),
            (2, code.count('2')),
            (3, code.count('3')),
            (4, code.count('4')),
            (5, code.count('5'))
        )
        syn = max(count, key=itemgetter(1))[0]
    code = status + str(syn)
    # print(code)
    covid = ['33', '03', '30', '23', '31', '32', '34']
    other = ['15', '05', '10', '25', '21', '22', '24', '01', '02', '04', '20']
    discard = ['11', '12', '13', '14', '35']
    if code in covid:
        return True
    elif code in other:
        return False
    else:
        return 'del'


def cleaner(path):
    print(f'Initial elements:\t{len(os.listdir(path))}')
    clear_ambiguous(path)
    print(f'Unambiguous elements:\t{len(os.listdir(path))}')
    files = os.listdir(path)
    for el in files:
        if label_func(el) == 'del':
            os.remove(os.path.join(path, el))
    print('Final elements: ', len(os.listdir(path)))
