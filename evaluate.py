
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

from functools import reduce

# ---------- 加载模型与数据 ----------
model = tf.keras.models.load_model("saved_models/cnn_lstm_attention_zscore_model.h5")

file_path = r"data/processed/补全.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)
pollutant_names = list(all_sheets.keys())
data_list = []
means, stds = {}, {}

for sheet_name, df in all_sheets.items():
    df = df.iloc[:, [0, 1, 4]]
    df.columns = ['date', 'hour', sheet_name]
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df[['datetime', sheet_name]]
    data_list.append(df)
    means[sheet_name] = df[sheet_name].mean()
    stds[sheet_name] = df[sheet_name].std()

merged_df = reduce(lambda left, right: pd.merge(left, right, on='datetime'), data_list)
merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

# ---------- 构造样本 ----------
merged_df['month'] = merged_df['datetime'].dt.month
train_df = merged_df[merged_df['month'].isin([1, 2, 3, 4])]
test_df = merged_df[merged_df['month'] == 5]

train_data = train_df[pollutant_names].values
test_data = test_df[pollutant_names].values

def create_multivariate_sequences(data, time_step=24):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

X_test, y_test = create_multivariate_sequences(test_data, 24)
X_test = X_test.reshape(-1, 24, len(pollutant_names), 1)

# ---------- 反归一化 ----------
def inverse_zscore(data, pollutant_names, means, stds):
    result = np.zeros_like(data)
    for i, name in enumerate(pollutant_names):
        result[:, i] = data[:, i] * stds[name] + means[name]
    return result

# ---------- 评估 ----------
y_pred = model.predict(X_test)
y_true_inv = inverse_zscore(y_test, pollutant_names, means, stds)
y_pred_inv = inverse_zscore(y_pred, pollutant_names, means, stds)

metrics = {"MSE": [], "RMSE": [], "MAE": [], "R²": []}
for i, name in enumerate(pollutant_names):
    true = y_true_inv[:, i]
    pred = y_pred_inv[:, i]
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    metrics["MSE"].append(mse)
    metrics["RMSE"].append(rmse)
    metrics["MAE"].append(mae)
    metrics["R²"].append(r2)

    print(f"【{name}】")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

# ---------- 可视化 ----------
plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics.keys()):
    plt.subplot(2, 2, i + 1)
    plt.bar(pollutant_names, metrics[metric], color='salmon')
    plt.title(metric)
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
plt.suptitle("各污染物预测性能指标（反归一化后）", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
