import numpy as np
import keras
from sklearn.model_selection import train_test_split

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

folder_names = np.load('model/folder_names.npy')
X = np.load('model/X_1.npy')
X.shape
y = np.load('model/y_1.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = keras.models.load_model('model/TSL_model.keras')
y_hat = model.predict(X_test)
folder_names[np.argmax(y_hat[4])]


y_true = np.argmax(y_test, axis=1)
y_hat = np.argmax(y_hat, axis=1)

multilabel_confusion_matrix(y_true, y_hat)

accuracy_score(y_true, y_hat) #%52