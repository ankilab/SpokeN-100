import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

#################### HELPER FUNCTIONS ####################
def get_audio_length(path):
    y, sr = librosa.load(path, sr=44100)
    return len(y)


# def _get_f0_frequency(path):
#     y, sr = librosa.load(path, sr=44100)
#     fft = np.fft.fft(y)
#     fft = np.abs(fft)
#     fft = fft[:len(fft) // 2]
#     freqs = np.fft.fftfreq(len(y), 1 / sr)
#     freqs = freqs[:len(freqs) // 2]
# 
#     # remove frequencies above 255 Hz
#     fft = fft[(freqs > 85) & (freqs < 255)]
#     freqs = freqs[(freqs > 85) & (freqs < 255)]
# 
#     # get frequency with highest amplitude --> f0 frequency
#     f0 = freqs[np.argmax(fft)]
#     return f0

def _get_f0_frequency(path):
    y, sr = librosa.load(path, sr=11025)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=512, n_thresholds=1)

    nan_indices = np.isnan(f0)
    f0 = f0[~nan_indices]
    voiced_flag = voiced_flag[~nan_indices]

    # filter out frequencies below 80 Hz and above 300 Hz
    voiced_flag = voiced_flag[(f0 < 300) & (f0 > 50)]
    f0 = f0[(f0 < 300) & (f0 > 50)]

    return np.mean(f0[voiced_flag])



def calculate_f0_frequency(folder: Path):
    files = [file for file in folder.iterdir() if file.is_file()]
    frequencies = []
    for file in files:
        try:
            frequency = _get_f0_frequency(file)
            frequencies.append((str(folder.name), int(file.stem), frequency))
            print(file, frequency)
        except Exception as e:
            print("Error in file: ", file)
            print(str(e))
    return frequencies
