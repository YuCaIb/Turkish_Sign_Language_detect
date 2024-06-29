import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from keras.src.callbacks import TensorBoard
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import keras

folder_names = np.load('model/folder_names.npy')
X = np.load('model/X_1.npy')
X.shape
y = np.load('model/y_1.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(layers.InputLayer(shape=(30, 1662)))
model.add(layers.LSTM(64, return_sequences=True, activation='relu'))
model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(folder_names.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='auto'),
             ModelCheckpoint(filepath='TSL_model.keras', monitor='val_loss', mode='min',
                             save_best_only=True, save_weights_only=False, verbose=1)]

model.summary()

# X_train.shape, y_train.shape, X_test.shape, y_test.shape
# array = np.random.rand(55)
# np.argmax(array)
# folder_names[np.argmax(array)]


# train
history = model.fit(x=X_train, y=y_train,
                    epochs=500, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[callbacks, tb_callback],
                    shuffle=False)




