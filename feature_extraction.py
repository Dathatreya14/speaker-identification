import os
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io.wavfile import read
import pickle

def calculate_delta(array):
    """Calculate and return the delta of the given feature vector matrix."""
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first = max(i - j, 0)
            second = min(i + j, rows - 1)
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio, rate):
    """Extract 20-dim MFCC features from an audio, perform CMS, and combine delta to make it a 40-dim feature vector."""    
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta)) 
    return combined

input_dir = "C:\\Users\\Muhammad Ramzan LLC\\Desktop\\draft code\\input audio\\person 3 audio files"
output_dir = "C:\\Users\\Muhammad Ramzan LLC\\Desktop\\draft code\\features\\person 3 features"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        audio_file_path = os.path.join(input_dir, filename)
        
        rate, audio = read(audio_file_path)

        feetures = extract_features(audio, rate)
        
        base_filename = os.path.splitext(filename)[0]
        npy_save_path = os.path.join(output_dir, f"{base_filename}.npy")
        pkl_save_path = os.path.join(output_dir, f"{base_filename}.pkl")
        
        np.save(npy_save_path, feetures)
        
        with open(pkl_save_path, "wb") as f:
            pickle.dump(feetures, f)
        
print("Features extracted and saved")


