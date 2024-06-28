from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical

label_map = {label: num for num, label in enumerate(holders)}