
import numpy as np
import pandas as pd
from functools import reduce
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# 读取并整合污染物数据
file_path = r"data/processed/补全.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)
pollutant_names = list(all_sheets.keys())
data_list = []

for sheet_name, df in all_sheets.items():
    df = df.iloc[:, [0, 1, 4]]
    df.columns = ['date', 'hour', sheet_name]
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df[['datetime', sheet_name]]
    data_list.append(df)

merged_df = reduce(lambda left, right: pd.merge(left, right, on='datetime'), data_list)
merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

# 划分训练和测试集
merged_df['month'] = merged_df['datetime'].dt.month
train_df = merged_df[merged_df['month'].isin([1, 2, 3, 4])]
test_df = merged_df[merged_df['month'] == 5]

train_data = train_df[pollutant_names].values
test_data = test_df[pollutant_names].values

# 构造时序样本
time_step = 24

def create_multivariate_sequences(data, time_step=24):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

X_train, y_train = create_multivariate_sequences(train_data, time_step)
X_train = X_train.reshape(-1, time_step, len(pollutant_names), 1)

# 模型构建
def build_model(time_step=24, feature_dim=6):
    inputs = tf.keras.Input(shape=(time_step, feature_dim, 1))
    x = layers.Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same')(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = layers.LSTM(32, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(32, return_sequences=True, dropout=0.3)(x)
    x = layers.MultiHeadAttention(num_heads=6, key_dim=4)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(feature_dim)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=['mae'])
    return model

model = build_model(time_step, len(pollutant_names))
model.fit(X_train, y_train, epochs=1200, batch_size=32, verbose=2)
model.save("saved_models/cnn_lstm_attention_zscore_model.h5")
print("Model trained and saved.")