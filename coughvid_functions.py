import os
import numpy as np
import pandas as pd
import librosa
import subprocess
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from scipy.stats import kurtosis
from scipy.integrate import simps
from scipy.signal import butter,filtfilt, cwt, hilbert, resample, decimate, spectrogram
from scipy.signal.windows import get_window


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


def preprocess_cough(x,fs, cutoff = 6000, normalize = True, filter_ = True, downsample = True):
    """
    Normalize, lowpass filter, and downsample cough samples in a given data folder 
    
    Inputs: x*: (float array) time series cough signal
    fs*: (int) sampling frequency of the cough signal in Hz
    cutoff: (int) cutoff frequency of lowpass filter
    normalize: (bool) normailzation on or off
    filter: (bool) filtering on or off
    downsample: (bool) downsampling on or off
    *: mandatory input
    
    Outputs: x: (float32 array) new preprocessed cough signal
    fs: (int) new sampling frequency
    """
    
    fs_downsample = cutoff*2
    
    #Preprocess Data
    if len(x.shape)>1:
        x = np.mean(x,axis=1)                          # Convert to mono
    if normalize:
        x = x/(np.max(np.abs(x))+1e-17)                # Norm to range between -1 to 1
    if filter_:
        b, a = butter(4, fs_downsample/fs, btype='lowpass') # 4th order butter lowpass filter
        x = filtfilt(b, a, x)
    if downsample:
        x = signal.decimate(x, int(fs/fs_downsample)) # Downsample for anti-aliasing
    
    fs_new = fs_downsample

    return np.float32(x), fs_new


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav
    folder: path to coughvid database and metadata_compiled csv"""
    
    df = pd.read_csv(folder + 'metadata_compiled.csv')
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if (counter%1000 == 0):
            print("Finished {0}/{1}".format(counter,len(names_to_convert)))
        if os.path.isfile(folder + name + '.webm'):
            subprocess.call(["ffmpeg", "-i", folder+name+".webm", folder+name+".wav"])
        elif os.path.isfile(folder + name + '.ogg'):
            subprocess.call(["ffmpeg", "-i", folder+name+".ogg", folder+name+".wav"])
        else:
            print("Error: No file name {0}".format(name))


#Use old segmentation
def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""
                
    cough_mask = np.array([False]*len(x))
    

    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

def compute_SNR(x, fs):
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    segments, cough_mask = segment_cough(x,fs)
    RMS_signal = 0 if len(x[cough_mask])==0 else np.sqrt(np.mean(np.square(x[cough_mask])))
    RMS_noise = np.sqrt(np.mean(np.square(x[~cough_mask])))
    SNR = 0 if (RMS_signal==0 or np.isnan(RMS_noise)) else 20*np.log10(RMS_signal/RMS_noise)
    return SNR


# Class that contains the feature computation functions 
class features:
    # output should be  a np.array 
    # names should be a list of the size of output
    # add the number of features in output of each function
    n_std_dev = 1
    n_dummy = 2
    n_EEPD = 19
    n_PRE = 1
    n_ZCR = 1
    n_RMSP = 1
    n_DF = 1
    n_spectral_features = 6
    n_SF_SSTD = 2
    n_MFCC = 26
    n_CF = 1
    n_LGTH = 1
    n_SSL_SD = 2
    
    def __init__(self, FREQ_CUTS):
        self.FREQ_CUTS = FREQ_CUTS # list of Frequency Bands for the PSD
        self.n_PSD = len(FREQ_CUTS)
        
    def std_dev(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['std_dev'] # list of output features  
        std_deviation = np.ones((1,1))*np.std(data[1])
        return std_deviation, names
    
    def dummy(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature (MUST BE AN ARRAY)
        names = ['dummy_feature_2','dummy_3']
        return np.array([1.,2.]), names
    
    def fft(self,data):
        """
        Compute the spectrum using FFT
        """
        fs, cough = data
        fftdata = np.fft.rfft(cough)
        return fftdata
    
    # Envelope Energy Peak Detection
    def EEPD(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = []
        fs,cough = data
        fNyq = fs/2
        nPeaks = []
        freq_step = 50
        for fcl in range(50,1000,freq_step):
            names = names + ['EEPD'+str(fcl)+'_'+str(fcl+freq_step)]
            fc = [fcl/fNyq, (fcl+50)/fNyq]
            b, a = butter(1, fc, btype='bandpass')
            bpFilt = filtfilt(b, a, cough)
            b,a = butter(2, 10/fNyq, btype='lowpass')
            eed = filtfilt(b, a, bpFilt**2)
            eed = eed/np.max(eed+1e-17)
            peaks,_ = signal.find_peaks(eed)
            nPeaks.append(peaks.shape[0])
        return np.array(nPeaks), names

    # Phase Power Ratio Estimation
    def PRE(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Power_Ratio_Est']
        fs,cough = data
        phaseLen = int(cough.shape[0]//3)
        P1 = cough[:phaseLen]
        P2 = cough[phaseLen:2*phaseLen]
        P3 = cough[2*phaseLen:]
        f = np.fft.fftfreq(phaseLen, 1/fs)
        P1 = np.abs(np.fft.fft(P1)[:phaseLen])
        P2 = np.abs(np.fft.fft(P2)[:phaseLen])
        P3 = np.abs(np.fft.fft(P3)[:phaseLen])
        P2norm = P2/(np.sum(P1)+1e-17)
        fBin = fs/(2*phaseLen +1e-17)
        f750,f1k,f2k5 = int(-(-750//fBin)), int(-(-1000//fBin)), int(-(-2500//fBin))
        ratio =  np.sum(P2norm[f1k:f2k5]) / np.sum(P2norm[:f750])
        return np.ones((1,1))*ratio, names
    
    # Zero Crossing Rate
    def ZCR(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Zero_Crossing_Rate']
        fs,cough = data
        ZCR = (np.sum(np.multiply(cough[0:-1],cough[1:])<0)/(len(cough)-1))
        return np.ones((1,1))*ZCR, names
    
    # RMS Power
    def RMSP(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['RMS_Power']
        fs,cough = data
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*RMS, names
    
    # Dominant Frequency
    def DF(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Dominant_Freq']
        fs,cough = data
        cough_fortan = np.asfortranarray(cough)
        freqs, psd = signal.welch(cough_fortan)
        DF = freqs[np.argmax(psd)]
        return  np.ones((1,1))*DF, names
    
    def spectral_features(self, data):
        names = ["Spectral_Centroid","Spectral_Rolloff","Spectral_Spread","Spectral_Skewness","Spectral_Kurtosis","Spectral_Bandwidth"]
        fs, x = data
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) # positive frequencies
        sum_mag = np.sum(magnitudes)
        
        # spectral centroid = weighted mean of frequencies wrt FFT value at each frequency
        spec_centroid = np.sum(magnitudes*freqs) / sum_mag

        #spectral roloff = frequency below which 95% of signal energy lies
        cumsum_mag = np.cumsum(magnitudes)
        spec_rolloff = np.min(np.where(cumsum_mag >= 0.95*sum_mag)[0]) 

        #spectral spread = weighted standard deviation of frequencies wrt FFT value
        spec_spread = np.sqrt(np.sum(((freqs-spec_centroid)**2)*magnitudes) / sum_mag)

        #spectral skewness = distribution of the spectrum around its mean
        spec_skewness = np.sum(((freqs-spec_centroid)**3)*magnitudes) / ((spec_spread**3)*sum_mag)

        #spectral kurtosis = flatness of spectrum around its mean
        spec_kurtosis =  np.sum(((freqs-spec_centroid)**4)*magnitudes) / ((spec_spread**4)*sum_mag)

        #spectral bandwidth = weighted spectral standard deviation
        p=2
        spec_bandwidth = (np.sum(magnitudes*(freqs-spec_centroid)**p))**(1/p)

        return np.array([spec_centroid, spec_rolloff, spec_spread, spec_skewness, spec_kurtosis, spec_bandwidth]), names
    
    # Spectral Flatness and spectral standard deviation
    def SF_SSTD(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Spectral_Flatness', 'Spectral_StDev']
        fs,sig = data
        nperseg = min(900,len(sig))
        noverlap = min(600,int(nperseg/2))
        freqs, psd = signal.welch(sig, fs, nperseg=nperseg, noverlap=noverlap)
        psd_len = len(psd)
        gmean = np.exp((1/psd_len)*np.sum(np.log(psd + 1e-17)))
        amean = (1/psd_len)*np.sum(psd)
        SF = gmean/amean
        SSTD = np.std(psd)
        return np.array([SF, SSTD]), names
        
    #Spectral Slope and Spectral Decrease
    def SSL_SD(self,data):
        names=['Spectral_Slope','Spectral_Decrease']
        b1=0
        b2=8000
        
        Fs, x = data
        s = np.absolute(np.fft.fft(x))
        s = s[:s.shape[0]//2]
        muS = np.mean(s)
        f = np.linspace(0,Fs/2,s.shape[0])
        muF = np.mean(f)

        bidx = np.where(np.logical_and(b1 <= f, f <= b2))
        slope = np.sum(((f-muF)*(s-muS))[bidx]) / np.sum((f[bidx]-muF)**2)

        k = bidx[0][1:]
        sb1 = s[bidx[0][0]]
        decrease = np.sum((s[k]-sb1)/(f[k]-1+1e-17)) / (np.sum(s[k]) + 1e-17)

        return np.array([slope, decrease]), names
    
    #MFCC
    def MFCC(self,data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of MFCC coefficient
        names = []; names_mean = []; names_std = []
        fs, cough = data
        n_mfcc = 13
        for i in range(n_mfcc):
            names_mean = names_mean + ['MFCC_mean'+str(i)]
            names_std = names_std +  ['MFCC_std'+str(i)]
        names = names_mean + names_std
        mfcc = librosa.feature.mfcc(y = cough, sr = fs, n_mfcc = n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        mfcc = np.append(mfcc_mean,mfcc_std)
        return mfcc, names
    
    # Crest Factor
    def CF(self,data):
        """
        Compute the crest factor of the signal
        """
        fs, cough = data
        peak = np.amax(np.absolute(cough))
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*peak/RMS, ['Crest_Factor']
    
    def LGTH(self,data):
        "Compute the length of the segment in seconds"
        fs, cough = data
        return np.ones((1,1))*(len(cough)/fs), ['Cough_Length']
    
    # Power spectral Density 
    def PSD(self,data):
        feat = []
        fs,sig = data
        nperseg = min(900,len(sig))
        noverlap=min(600,int(nperseg/2))
        freqs, psd = signal.welch(sig, fs, nperseg=nperseg, noverlap=noverlap)
        dx_freq = freqs[1]-freqs[0]
        total_power = simps(psd, dx=dx_freq)
        for lf, hf in self.FREQ_CUTS:
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            band_power = simps(psd[idx_band], dx=dx_freq)
            feat.append(band_power/total_power)
        feat = np.array(feat)
        feat_names = [f'PSD_{lf}-{hf}' for lf, hf in self.FREQ_CUTS]
        return feat, feat_names
    