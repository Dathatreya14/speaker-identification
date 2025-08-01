import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

#for each speaker, set the source to the directory as the speakers audio dataset directory
source = "it consists of the directory of the npy files"

dest = "directory where the the drained gmm files will get stored"

feature_files = [f for f in os.listdir(source) if f.endswith('.npy') or f.endswith('.pkl')]

features = np.asarray(())

for file in feature_files:
    if file.endswith('.npy'):
        feature_vector = np.load(os.path.join(source, file))
    elif file.endswith('.pkl'):
        with open(os.path.join(source, file), 'rb') as f:
            feature_vector = pickle.load(f)

    if features.size == 0:
        features = feature_vector
    else:
        features = np.vstack((features, feature_vector))


gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
gmm.fit(features)


speaker_name = file.split("_")[0]  
gmm_filename = os.path.join(dest, f"{speaker_name}.gmm")

with open(gmm_filename, 'wb') as f:
    pickle.dump(gmm, f)

print(f"Model trained and saved for speaker: {speaker_name}")
