# CNN-LSTM-ML
"""
# Air Quality Prediction using CNN-LSTM-Attention

This project predicts multivariate air pollution data (e.g., PM2.5, PM10, O3, NO2, SO2, CO) using a hybrid deep learning model combining CNN, LSTM, and Multi-Head Attention. The model supports hourly-level prediction based on historical data.

## 📁 Project Structure

```
AirQuality-Prediction-CNN-LSTM-ML/
├── data/                  # Contains raw/processed Excel data
├── src/                   # Python source scripts
│   ├── preprocess.py
│   ├── model_train.py
│   ├── evaluate.py
├── saved_models/          # Trained Keras model
├── notebooks/             # (Optional) Jupyter notebooks
├── requirements.txt       # Required libraries
├── README.md              # Project description
```

## 🔧 Usage

```bash
# Train model
python src/model_train.py

# Evaluate model
python src/evaluate.py
```

## 📈 Results

- RMSE (PM2.5): ~4.1
- R² (NO2): ~0.91

## ✅ Features
- Data interpolation & normalization
- Multivariate sequence construction
- Hybrid CNN + LSTM + Attention deep model
- Performance evaluation and visualization

## 📜 License

MIT License
"""
