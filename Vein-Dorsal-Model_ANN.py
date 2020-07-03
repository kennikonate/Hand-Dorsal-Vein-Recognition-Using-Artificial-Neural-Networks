
# Installing Tensorflow
# Installing Keras
# pip install --upgrade keras

# Importing the libraries

import pandas as pd

# Importing the dataset
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
root= tk.Tk()
def getEXC():
    global df  
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path)
    return df

root.mainloop()
data=getEXC()

X = data.iloc[:, :109].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_108 = LabelEncoder()
X[:, 108]= labelencoder_X_108.fit_transform(X[:, 108])
onehotencoder = OneHotEncoder(categorical_features = [108])
X= onehotencoder.fit_transform(X).toarray()
y=X[:, :25]
X=X[:, 25:]

#Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 108, init = 'uniform', input_dim = 108))
#
## Adding the first hidden layer
classifier.add(Dense(output_dim = 101, init = 'uniform', activation = 'tanh'))
#
# Adding the output layer
classifier.add(Dense(output_dim = 25))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss='mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
