import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from feature_extraction import extract_features
import time

source = "C:\\Users\\Muhammad Ramzan LLC\\Desktop\\draft code\\features"
modelpath = "C:\\Users\\Muhammad Ramzan LLC\\Desktop\\draft code\\gmm files"


gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [os.path.basename(fname).split(".gmm")[0] for fname in gmm_files]

print("Enter the File name from Test Audio Sample Collection:")
path = input().strip()  
print(f"Testing Audio: {path}")

audio_file_path = os.path.join(source, path)

sr, audio = read(audio_file_path)

vector = extract_features(audio, sr)

log_likelihood = np.zeros(len(models))
for i in range(len(models)):
    gmm = models[i]
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()
    
print("\nScores for each speaker:")
for i in range(len(speakers)):
    print(f"{gmm_files[i]}: {log_likelihood[i]}")


max_score_index = np.argmax(log_likelihood)
winner_speaker = speakers[max_score_index]

if winner_speaker == "Audio001":
    print("datha is the speaker.")
elif winner_speaker == "ryanvoice":
    print("ryan is the speaker.")
elif winner_speaker == "Vr9":
    print("hima is the speaker.")
else:
    print("Unknown speaker identified.")

# End the process
print("...")
