# %%

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping
import os
import datetime
import time



df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

df = df1

df1 = pd.concat([df["turbidity"], df["ph"], df["alkalinity"], df["conductivity"], df["cl"], df["temperature"], df["river"], df["2-MIB"]], axis=1)
df1 = pd.DataFrame(df1)

df = df2

df2 = pd.concat([df["turbidity"], df["ph"], df["alkalinity"], df["conductivity"], df["cl"], df["temperature"], df["river"], df["2-MIB"]], axis=1)
df2 = pd.DataFrame(df2)

df1 = df1.dropna()
df2 = df2.dropna()



train = df1
x_train = pd.concat([train["turbidity"], train["ph"], train["alkalinity"], train["conductivity"], train["cl"], train["temperature"], train["river"]], axis=1)
x_train = pd.DataFrame(x_train)
y_train = pd.concat([train["2-MIB"]])
y_train = pd.DataFrame(y_train)



x_mean = x_train.mean()
x_std = x_train.std()


x_train = (x_train - x_mean) / x_std



train = []
train = df2

x_test = pd.concat([train["turbidity"], train["ph"], train["alkalinity"], train["conductivity"], train["cl"], train["temperature"], train["river"]], axis=1)
x_test = pd.DataFrame(x_test)
y_test = pd.concat([train["2-MIB"]])
y_test = pd.DataFrame(y_test)

x_test = (x_test - x_mean) / x_std




early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

def step_decay(epoch):
    x = 0.001
    if epoch >= 50: x = 0.0001
    if epoch >= 100: x = 0.00001
    if epoch >= 300: x = 0.000001
    return x


lr_decay = LearningRateScheduler(step_decay)

# モデルの構築
n_in = 7
n_hidden = 1024
n_out = 1
epochs = 50
batch_size = 32

acti = 'tanh'

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in, activation=acti))
model.add(Dense(n_hidden, activation=acti))
model.add(Dense(n_hidden, activation=acti))
model.add(Dense(n_hidden, activation=acti))
model.add(Dense(n_hidden, activation=acti))
model.add(Dense(units=n_out))


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)


history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    # callbacks=[lr_decay]
                    )


new_folder = "2-MIB" + "_node" + str(n_hidden) + "_ep" + str(epochs)
if not os.path.exists(new_folder): 
    os.mkdir(new_folder)

train_predict = model.predict(x_train)

Train_predict = pd.DataFrame(train_predict);
Train_predict.to_csv(new_folder + '/' + new_folder + '_train_predict.csv');

test_predict = model.predict(x_test)

Test_predict = pd.DataFrame(test_predict);
Test_predict.to_csv(new_folder + '/' + new_folder + '_test_predict.csv');

Y_train = pd.DataFrame(y_train);
Y_train.to_csv(new_folder + '/' + new_folder + '_y_train.csv');

Y_test = pd.DataFrame(y_test);
Y_test.to_csv(new_folder + '/' + new_folder + '_y_test.csv');

plt.scatter(Y_train, train_predict, label='train', color='black')
plt.legend(loc='upper left')
plt.title("train")
plt.savefig(new_folder + '/' + new_folder + '_FFNN_train.pdf')
plt.show()

plt.scatter(Y_test, test_predict, label='test', color='black')
plt.legend(loc='upper right')
plt.title("test")
plt.savefig(new_folder + '/' + new_folder + '_FFNN_test.pdf')
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(new_folder + '/' + new_folder + '_LSTM_epochs.pdf')
plt.show()


# モデルの保存
open(new_folder + '/model.json', "w").write(model.to_json())

model.save(new_folder + '/model.h5')


