
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import os

# CSVファイルから訓練データとテストデータを読み込む
df_train = pd.read_csv('カビ臭/csv/train.csv')
df_test = pd.read_csv('カビ臭/csv/test.csv')

# 必要な列のみを選択し、NaNを削除
df_train = df_train[['turbidity', 'pH', 'alkalinity', 'conductivity', 'cl', 'temperature', 'river', '2-MIB']].dropna()
df_test = df_test[['turbidity', 'pH', 'alkalinity', 'conductivity', 'cl', 'temperature', 'river', '2-MIB']].dropna()

# 訓練データとテストデータの特徴量とターゲットを準備
x_train = df_train.drop('2-MIB', axis=1)
y_train = df_train[['2-MIB']]
x_test = df_test.drop('2-MIB', axis=1)
y_test = df_test[['2-MIB']]

# 特徴量の標準化
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# 早期停止の設定
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 学習率のスケジューリング関数
def step_decay(epoch):
    x = 0.001
    if epoch >= 50: x = 0.0001
    if epoch >= 100: x = 0.00001
    if epoch >= 300: x = 0.000001
    return x

lr_decay = LearningRateScheduler(step_decay)

# ニューラルネットワークのモデルを構築
n_in = 7
n_hidden = 64
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




model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# モデルの訓練
history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=2, 
                    validation_data=(x_test, y_test), 
                    callbacks=[early_stopping])

# 結果の保存ディレクトリを作成
new_folder = "2-MIB_node16_epochs2"
os.makedirs(new_folder, exist_ok=True)

# 訓練データとテストデータの予測値を保存
pd.DataFrame(model.predict(x_train)).to_csv(f'{new_folder}/train_predict.csv')
pd.DataFrame(model.predict(x_test)).to_csv(f'{new_folder}/test_predict.csv')

# 実際のターゲット値を保存
y_train.to_csv(f'{new_folder}/y_train.csv')
y_test.to_csv(f'{new_folder}/y_test.csv')

# 訓練データとテストデータの予測結果をプロット
plt.scatter(y_train, model.predict(x_train), label='train', color='black')
plt.legend(loc='upper left')
plt.title("train")
plt.savefig(f'{new_folder}/FFNN_train.pdf')
plt.show()

plt.scatter(y_test, model.predict(x_test), label='test', color='black')
plt.legend(loc='upper right')
plt.title("test")
plt.savefig(f'{new_folder}/FFNN_test.pdf')
plt.show()

# 損失と検証損失をエポックごとにプロット
plt.plot(history.history['loss'], marker='.', label='loss')
plt.plot(history.history['val_loss'], marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(f'{new_folder}/LSTM_epochs.pdf')
plt.show()

# モデルを保存
model.save(f'{new_folder}/model.h5')
