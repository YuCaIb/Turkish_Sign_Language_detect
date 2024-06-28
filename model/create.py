from keras import Sequential
from keras import layers
from ..words import holders

model = Sequential()
model.add(layers.LSTM(128, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(layers.Dense(256, return_sequences=True, activation='relu'))
model.add(layers.Dense(128, return_sequences=False, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(holders.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


















# keras.Sequential
# keras.layers.LSTM
# keras.layers.Dense
# keras.callbacks.TensorBoard
