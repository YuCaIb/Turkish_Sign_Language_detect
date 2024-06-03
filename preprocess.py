from tensorflow.keras.utils import to_categorical
import numpy as np
import os

from collect_data import holders
from utilty import DATA_PATH, sequence_length

label_map = {label:num for num, label in enumerate(holders)}


sequences, labels = [], []
for action in holders:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)