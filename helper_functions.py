import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
from scipy.io import wavfile
from segmentation import segment_cough

def plot_sample(path_wav, mask=False):
    data, sample_rate = librosa.load(path=path_wav)
    plt.figure(figsize=(10,5))
    if mask:
        _, cough_mask = segment_cough(data, sample_rate)
        time = np.linspace(0, len(data)/sample_rate, num=len(data))
        plt.plot(time, data)
        plt.plot(time, cough_mask)
    else:
        librosa.display.waveshow(data, sr=sample_rate)
    plt.xticks(np.arange(0, len(data)/sample_rate, step=1))
    plt.xlabel('Time (s)')
    plt.show()

def plot_multiple_samples(path_wavs, max=12, mask=False):
    plots = []
    fig, axs = plt.subplots(max//4, 4, figsize=(20, max))
    index, counter = [0, 0], 0
    for file in os.listdir(path_wavs):
        if counter < max:
            if file.endswith('.wav'):
                data, sample_rate = librosa.load(f'data/coughvid/{file}')
                if mask:
                    _, cough_mask = segment_cough(
                        data, sample_rate,
                        cough_padding=0.2,
                        min_cough_len=0.1,
                        th_l_multiplier = 0.1,
                        th_h_multiplier = 2
                    )
                    time = np.linspace(0, len(data)/sample_rate, num=len(data))
                    axs[index[0], index[1]].plot(time, data)
                    axs[index[0], index[1]].plot(time, cough_mask)
                    axs[index[0], index[1]].set_title(f'{index} - {len(plots)}')
                else:
                    librosa.display.waveshow(data, sr=sample_rate)
                plt.xticks(np.arange(0, len(data)/sample_rate, step=1))
                plt.xlabel('Time (s)')
                print(index, file, len(plots))
                plots.append(file)
                index[1] += 1
                if index[1] == 4:
                    index[0] += 1
                    index[1] = 0
                counter += 1
    plt.show()
    return plots

def plot_spectrum(path_wav, scipy=False):
    if scipy:
        Fs, aud  = wavfile.read(path_wav)
        plt.figure(figsize=(10,10))
        _, _, _, _ = plt.specgram(aud, Fs=Fs)
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
    else:
        data, sample_rate = librosa.load(path=path_wav)
        d = librosa.stft(data)
        D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
        fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,10))
        img = librosa.display.specshow(D, y_axis='linear', x_axis='s',sr=sample_rate,ax=ax[0])
        ax[0].set(title='Linear frequency power spectrogram')
        ax[0].label_outer()
        librosa.display.specshow(D,y_axis='log',x_axis='s',sr=sample_rate,ax=ax[1])
        ax[1].set(title='Log frequency power spectrogram')
        ax[1].label_outer()
        fig.colorbar(img, ax=ax, format='%+2.f dB')
    plt.show()

def plot_features(path_wav, feature):
    data, sample_rate = librosa.load(path=path_wav)
    if feature == 'chroma':
        C = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=C, sr=sample_rate)
        fig, ax = plt.subplots(figsize=(10,6))
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='s', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title='Chromagram')
    elif feature == 'tempo':
        oenv = librosa.onset.onset_strength(y=data, sr=sample_rate)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate)
        # Compute global onset autocorrelation
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)
        # Estimate the global tempo for display purposes
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sample_rate)[0]
        fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
        times = librosa.times_like(oenv, sr=sample_rate)
        ax[0].plot(times, oenv, label='Onset strength')
        ax[0].label_outer()
        ax[0].legend(frameon=True)
        librosa.display.specshow(tempogram, sr=sample_rate,x_axis='s', 
            y_axis='tempo', cmap='magma',ax=ax[1])
        ax[1].axhline(tempo, color='g', linestyle='--', alpha=1,
                    label='Estimated tempo={:g}'.format(tempo))
        ax[1].legend(loc='upper right')
        ax[1].set(title='Tempogram')
    plt.show()