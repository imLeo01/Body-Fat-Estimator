import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. Load dataset ---
df = pd.read_csv('ANN/BTVN/bodyfat.csv')

# Xem nhanh cấu trúc dữ liệu
print(df.head())
print(df.columns)
print(df.columns.tolist())

# Giả sử các cột tiêu biểu là: %BodyFat (target), Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist
# Nếu tên cột khác, bạn sửa lại cho đúng
feature_cols = [
    'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen',
    'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'
]
target_col = 'BodyFat'  # sửa nếu tên khác

# Lọc lấy các cột cần thiết, loại bỏ missing
df2 = df[feature_cols + [target_col]].dropna()

X = df2[feature_cols].values
y = df2[target_col].values

# --- 2. Chuẩn hóa dữ liệu ---
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- 3. Chia train / test ---
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

# --- 4. Xây mô hình ANN ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# --- 5. Huấn luyện ---
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=600,
                    batch_size=16,
                    verbose=1)

# --- 6. Đánh giá trên test set ---
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE = {mse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")

# Nếu muốn lưu model & scaler:
model.save("bodyfat_ann_model.h5")
import pickle
with open("bodyfat_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
