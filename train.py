import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from preprocess import X, y

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# complie
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# train
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
# model_summary()

# model.save('TSL_v1.h5')
