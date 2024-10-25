import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

# Đọc dữ liệu
data = pd.read_csv('weather_data.csv')  # Thay đổi đường dẫn tới tệp CSV của bạn

# Chuyển đổi cột province thành dạng số bằng One-Hot Encoding
data = pd.get_dummies(data, columns=['province'], drop_first=True)
data = pd.get_dummies(data, columns=['wind_d'], drop_first=True)

# Khởi tạo dictionary để lưu mô hình cho từng tỉnh
models = {}

# Tạo mô hình cho từng tỉnh nếu mô hình chưa tồn tại
if not os.path.exists('models/weather_rain_prediction_models.pkl'):
    for province in data.columns[data.columns.str.startswith('province_')]:
        province_data = data[data[province] == 1].copy()
        province_data = province_data.drop(columns=['date', province])

        X = province_data.drop(columns=['rain'])
        y = province_data['rain']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

        models[province] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }

    with open('models/weather_rain_prediction_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("Tất cả mô hình đã được lưu.")

# Dự đoán từ input của người dùng
def predict_rain():
    # Nhập thông tin từ người dùng
    province = input("Nhập tên tỉnh: ")
    max_temp = float(input("Nhập nhiệt độ tối đa: "))
    min_temp = float(input("Nhập nhiệt độ tối thiểu: "))
    wind_speed = float(input("Nhập tốc độ gió: "))
    humidi = float(input("Nhập độ ẩm: "))
    cloud = float(input("Nhập độ che phủ mây: "))
    pressure = float(input("Nhập áp suất: "))
    wind_direction = input("Nhập hướng gió (ví dụ: NNE): ")

    # Tạo DataFrame cho dữ liệu đầu vào
    input_data = pd.DataFrame([[max_temp, min_temp, wind_speed, humidi, cloud, pressure]], 
                               columns=['max', 'min', 'wind', 'humidi', 'cloud', 'pressure'])

    # Nạp mô hình từ file
    with open('models/weather_rain_prediction_models.pkl', 'rb') as f:
        models = pickle.load(f)

    province_columns = [f'province_{province}']
    if province_columns[0] in data.columns:
        model_info = models[province_columns[0]]
        model = model_info['model']
        scaler_X = model_info['scaler_X']
        scaler_y = model_info['scaler_y']

        # Thêm các cột wind_d đã được One-Hot Encoding vào input_data
        wind_direction_columns = [f'wind_d_{d}' for d in data['wind_d'].unique() if d != data['wind_d'].unique()[0]]
        for column in wind_direction_columns:
            input_data[column] = 0  # Thêm cột với giá trị mặc định là 0
        input_data[f'wind_d_{wind_direction}'] = 1  # Gán giá trị cho hướng gió tương ứng

        input_scaled = scaler_X.transform(input_data)
        input_scaled = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))

        predicted_rain_scaled = model.predict(input_scaled)
        predicted_rain = scaler_y.inverse_transform(predicted_rain_scaled)
        print(f"Dự đoán lượng mưa cho tỉnh {province}: {predicted_rain.flatten()[0]:.2f} mm")
    else:
        print("Tỉnh không hợp lệ.")

# Gọi hàm dự đoán
predict_rain()
