import os, sys
sys.path.append(os.path.abspath('../src'))

import numpy as np
from modules.coughvid_functions import preprocess_cough, features


def classify_cough(x, fs, model, scaler):
    """Classify whether an inputted signal is a cough or not using filtering, feature extraction, and ML classification
    Inputs: 
        x: (float array) raw cough signal
        fs: (int) sampling rate of raw signal
        model: cough classification ML model loaded from file
    Outputs:
        result: (float) probability that a given file is a cough 
    """
    try: 
        x,fs = preprocess_cough(x,fs)
        data = (fs,x)
        FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
        features_fct_list = ['EEPD','ZCR','RMSP','DF','spectral_features','SF_SSTD','SSL_SD','MFCC','CF','LGTH','PSD']
        feature_values_vec = []
        obj = features(FREQ_CUTS)
        for feature in features_fct_list:
            feature_values, feature_names = getattr(obj,feature)(data)
            for value  in feature_values:
                if isinstance(value,np.ndarray):
                    feature_values_vec.append(value[0])
                else:
                    feature_values_vec.append(value)
        feature_values_scaled = scaler.transform(np.array(feature_values_vec).reshape(1,-1))
        result = model.predict_proba(feature_values_scaled)[:,1]
        return result[0]
    except:
        "Feature extraction fails when the audio is completely silent"
        return 0