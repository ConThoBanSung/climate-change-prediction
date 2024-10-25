import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu
data = pd.read_csv('weather_data.csv')  # Thay đổi đường dẫn tới tệp CSV của bạn
print(data[data['rain'] < 0])
