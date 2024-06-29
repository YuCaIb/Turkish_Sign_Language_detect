import os.path
import numpy as np

from keras.src.utils import to_categorical
from pathlib import Path

seq_len = 30


def get_folder_names(directory_path):
    """collects folders names in to a list"""
    home = Path.home()
    pathy = home / directory_path
    folders = [folder.name for folder in pathy.iterdir() if folder.is_dir()]
    return folders


path = 'Desktop/Turkish_Sign_Language/collection'

folder_names = get_folder_names(path)

home_dir = Path.home()
final_path = home_dir / path

label_map = {label: num for num, label in enumerate(folder_names)}

sequences, labels = [], []
for folder in folder_names:
    for sequence in np.array(os.listdir(os.path.join(final_path, folder))).astype(int):
        window = []
        for frame_num in range(seq_len):
            res = np.load(os.path.join(final_path, folder, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[folder])

# train test split
X = np.array(sequences)
X.shape  # (1650, 30, 1662)
# type(X[54][29][1661]) float64
np.save("X_1.", X)
y = to_categorical(labels).astype(int)
y.shape  # (1650, 55)
np.save("y_1.npy", y)

# sequences, labels = [], []
# for folder_name in folder_names:
#     folder_path = os.path.join(final_path, folder_name)  # Precompute folder path
#
#     # Load all sequences for the folder at once (vectorized loading)
#     sequences_in_folder = np.stack([
#         np.load(os.path.join(folder_path, str(sequence), '{}.npy'.format(frame_num)))
#         for sequence in range(seq_len)
#         for frame_num in range(seq_len)
#     ], axis=0)
#
#     sequences.append(sequences_in_folder)
#     labels.append(label_map[folder_name])


# folder_names= np.array(folder_names)
# np.save('folder_names',folder_names)


# Faster Version
# sequences = []
# labels = []
#
# for folder in folder_names:
#     folder_path = os.path.join(final_path, folder)
#     sequence_paths = sorted(os.listdir(folder_path), key=int)  # Ensure sequences are sorted numerically
#
#     sequences.extend([
#         [np.load(os.path.join(folder_path, str(sequence), f"{frame_num}.npy")) for frame_num in range(seq_len)]
#         for sequence in sequence_paths
#     ])
#
#     labels.extend([label_map[folder]] * len(sequence_paths))

# train test split
# X = np.array(sequences)
# X.shape
# type(X[54][29][1661])
# np.save("X_2.", X)
# y = to_categorical(labels).astype(int)
# y.shape
# np.save("y_2.npy", y)
