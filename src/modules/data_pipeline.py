import os
import librosa
import numpy as np
from PIL import Image
from joblib import load
from scipy.io import wavfile
from fastai.vision.all import *
from xgboost import XGBClassifier
from modules.coughvid_functions import classify_cough
from modules.helper_functions import (
    segment_cough,
    preprocess_cough,
    classify_cough,
)

import warnings
warnings.filterwarnings('ignore') 


def check(model_dir, audio_path):
    # Load model and scaler
    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(model_dir, 'model.pkl'))
    xgb_scaler = load(os.path.join(model_dir, 'scaler.pkl'))
    rate, audio = wavfile.read(audio_path)
    probability = classify_cough(audio, rate, xgb_model, xgb_scaler)
    return probability


def process(path, save=True, mkdir=False, display=False):
    images = []
    audio, rate = librosa.load(path)
    segments, _ = segment_cough(audio, rate)
    if len(segments) > 0:
        for segment in segments:
            prepro_segment, _ = preprocess_cough(segment, rate)
            spectrum = librosa.stft(prepro_segment)
            spectrum = spectrum.astype(np.uint8)
            img = Image.fromarray(spectrum)
            images.append(img)
            if display:
                img.show(img)
    if save:
        cache_path = path[:path.rindex('/')]
        if mkdir:
            cache_path = os.path.join(path[:path.rindex('/')], 'cache')
            if not os.path.exists(cache_path):
                os.mkdir(cache_path)
            else:
                for file in os.listdir(cache_path):
                    os.remove(os.path.join(cache_path, file))
        for i, img in enumerate(images):
            img.save(os.path.join(cache_path, f'{i}.png'))
    return images


def label_func(file):
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
    covid = ['33', '03', '30', '23', '31', '32', '34']
    other = ['15', '05', '10', '25', '21', '22', '24', '01', '02', '04', '20']
    if code in covid:
        return True
    elif code in other:
        return False


def predict(model_path, data_path, clean=True, verbose=False):
    predictions = []
    model = load_learner(Path(model_path))
    if verbose:
        print('LEGEND\n')
        print('index: [0]  label:', model.dls.vocab[0], ' = NON-COVID19')
        print('index: [1]  label:', model.dls.vocab[1], '  = COVID19')
        print('\n')
    for img in os.listdir(data_path):
        if img.endswith('.png'):
            prediction = model.predict(os.path.join(data_path, img))
            if verbose: print(img, prediction)
            if clean:
                predictions.append(prediction[2][1].item())
            else:
                predictions.append(prediction)
    # predictions: list of decimal probabilities of having COVID19-related cough
    return predictions


def pipeline(
    data_directory,
    xgb_model_path,
    cnn_model_path,
    check_threshold=0.8,
    predict_threshold=0.8,
    weighing_formula='majority_ceil',
    verbose=False
    ):
    audio_path = os.path.join(
        data_directory,
        [
            el for el in os.listdir(data_directory)
            if not os.path.isdir(el)
        ][0]
    )
    # check if the sample contains cough
    check_probability = check(xgb_model_path, audio_path)
    # if check_threshold > 0:
    #     if check_probability < check_threshold:
    #         return 'NA' # no cough
    # process the sample
    process(audio_path)
    # predict the sample
    cache_path = os.path.join(data_directory)
    cnn_model_path = os.path.join(cnn_model_path, 'model.pkl')
    predictions = predict(cnn_model_path, cache_path, verbose=verbose)
    print(predictions)
    print('\n')
    if weighing_formula == 'average_floor':
        prediction = int(np.floor(np.average(predictions)))
    elif weighing_formula == 'average_ceil':
        prediction = int(np.ceil(np.average(predictions)))
    elif weighing_formula == 'majority_floor':
        prediction = int(np.floor(np.average(
            [1 if el > predict_threshold else 0 for el in predictions]
        )))
    elif weighing_formula == 'majority_ceil':
        prediction = int(np.ceil(np.average(
            [1 if el > predict_threshold else 0 for el in predictions]
        )))
    return prediction, np.average(predictions), check_probability
