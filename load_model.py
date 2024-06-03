from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense








model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# load
model.load_weights('action.h5')


#Accuracy Score:
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
