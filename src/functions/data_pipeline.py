from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED
from helper_functions import segment_cough, preprocess_cough, classify_cough, label_func
from coughvid_functions import classify_cough
from xgboost import XGBClassifier
from scipy.io import wavfile
from fastai.vision.all import *
from joblib import load
from PIL import Image
import numpy as np
import librosa
import os

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


def process(path, save=True, display=False):
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
        cache_path = os.path.join(path[:path.rindex('/')], 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        else:
            for file in os.listdir(cache_path):
                os.remove(os.path.join(cache_path, file))
        for i, img in enumerate(images):
            img.save(os.path.join(cache_path, f'{i}.png'))
    return images


def predict(model_path, data_path, clean=True, verbose=False):
    predictions = []
    model = load_learner(Path(model_path))
    if verbose:
        print('LEGEND\n')
        print('index: [0]  label:', model.dls.vocab[0], ' = NON-COVID19')
        print('index: [1]  label:', model.dls.vocab[1], '  = COVID19')
        print('\n')
    for img in os.listdir(data_path):
        prediction = model.predict(os.path.join(data_path, img))
        if verbose: print(img, prediction)
        if clean:
            predictions.append(prediction[2][0].item())
        else:
            predictions.append(prediction)
    return predictions


def pipeline(
    data_directory,
    xgb_model_path,
    cnn_model_path,
    check_threshold=0.8,
    predict_threshold=0.8,
    weighing_formula='average'
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
    if check_probability < check_threshold:
        return 0 # no cough
    # process the sample
    process(audio_path)
    # predict the sample
    cache_path = os.path.join(data_directory, 'cache')
    cnn_model_path = os.path.join(cnn_model_path, 'model.pkl')
    predictions = predict(cnn_model_path, cache_path)
    if weighing_formula == 'average_floor':
        prediction = int(np.floor(np.average(predictions)))
    elif weighing_formula == 'average_ceil':
        prediction = int(np.ceil(np.average(predictions)))
    elif weighing_formula == 'majority_floor':
        predictions = [1 if el > predict_threshold else 0 for el in predictions]
        prediction = int(np.floor(np.average(predictions)))
    elif weighing_formula == 'majority_ceil':
        predictions = [1 if el > predict_threshold else 0 for el in predictions]
        prediction = int(np.ceil(np.average(predictions)))
    # return prediction
    print(prediction)



pipeline('../trials', '../models/xgb', '../models/cnn', weighing_formula='average_ceil')
pipeline('../trials', '../models/xgb', '../models/cnn', weighing_formula='average_floor')
pipeline('../trials', '../models/xgb', '../models/cnn', weighing_formula='majority_ceil')
pipeline('../trials', '../models/xgb', '../models/cnn', weighing_formula='majority_floor')
