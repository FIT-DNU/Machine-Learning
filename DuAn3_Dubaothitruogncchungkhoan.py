import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
f = pd.read_csv(
    "/content/drive/MyDrive/predict_stock/vnindex_historical_Mar2015_Mar2023.csv",index_col=0)
# Chuyển đổi chỉ mục (index) thành định dạng datetime
df.index = pd.to_datetime(df.index)
# Sắp xếp lại index tăng dần theo date
df = df.sort_index()
df.head(5)
df.columns
Kết quả
	Index(['Open', 'High', 'Low', 'Volume', 'Close'], dtype='object')

# Kiểm tra chi tiết dữ liệu
print(f'Tập dữ liệu chứa {df.shape[0]} dòng và {df.shape[1]} cột')
df.info()
# Điền những giá trị còn thiếu trong column 'Volume' bằng mode
df['Volume'].fillna(df['Volume'].mode()[0], inplace=True)

# Kiểm tra lại giá trị còn trống (null)
df.isna().sum()
print(df.describe())
data = df.copy()

# Giá đóng cửa cửa của i ngày trước
for i in range(30):
  data[f'Close_{i}'] = data['Close'].shift(i)

# Giá đóng cửa của ngày hôm sau
data[f'Tomorrow'] = data['Close'].shift(-1)

data = data.dropna(axis=0)
data
# Sử dụng hàm corr() để thấy sự tương quan giữa các biến dữ liệu
head_df = data.corr()

# Sử dụng histplot để trực quan hóa mối quan hệ giữa các biến
plt.figure(figsize=(12,8))
sns.heatmap(head_df, linewidths=0.5)
# Gắn nhãn cho tập dữ liệu
# Dữ liệu đặc trưng, biến độc lập
x = df.iloc[:,:-1]
# Biến phụ thuộc
y = df['Tomorrow']
# Chia tập train, test để dùng huấn luyện và đánh giá
from sklearn.model_selection import train_test_split
# Gọi model LinearRegression
from sklearn.linear_model import LinearRegression
# Gọi model RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Gọi model Support vector machine regressor (SVR)
from sklearn.svm import SVR
# Các metrics dùng để đánh giá mô hình
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
# Chia tập dữ liệu train, test
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.3)
# Xây dựng mô hình Random ForestRegressor
Rgs = RandomForestRegressor(n_estimators= 100, random_state = 0).fit(x_train, y_train)
y_pred_rgs = Rgs.predict(x_test)

# Xây dựng mô hình SVR
svr = SVR().fit(x_train, y_train)
y_pred_svr = svr.predict(x_test)

# Xây dựng mô hình Linear Regression
lgr = LinearRegression().fit(x_train, y_train)
y_pred_lgr = lgr.predict(x_test)
# Định nghĩa hàm in ra kết quả đánh giá mô hình
def print_evaluate(y_test, y_pred, string):
  mae = mean_absolute_error(y_test, y_pred).round(4)
  mse = mean_squared_error(y_test, y_pred).round(4)
  r2_squared = r2_score(y_test, y_pred).round(4)

  print(f'Đánh giá kết quả của {string}')
  print(f'MAE: {mae}')
  print(f'MSE: {mse}')
  print(f'r2_score: {r2_squared}')
  print('-----------------------------')
# Định nghĩa hàm plot để trực quan hóa giá trị actual và predict
def plot_validation(y_test, y_pred, string):
  # Tạo dataframe cho y_test
  data = pd.DataFrame(y_test)
# Gắn thêm column giá trị dự đoán vào data
  data['pred'] = y_pred
# Đổi lại tên column
  data.columns = ['actual', 'pred']
# Sắp xếp lại giá trị index theo thứ tự
  data = data.sort_index()
# Vẽ biểu đồ của giá dự đoán và giá thực tế
  plt.figure(figsize=(14,8))
  plt.plot(data[['actual', 'pred']])
  plt.title(f'{string}', fontsize=20)
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Giá', fontsize=18)
# kết quả đánh giá của module RandomForest
print_evaluate(y_test,y_pred_rgs, 'Random Forest')

# kết quả đánh giá của module SVR
print_evaluate(y_test, y_pred_svr, 'SVR')

# kết quả đánh giá của module Linear Regression
print_evaluate(y_test, y_pred_lgr, 'Linear Regression')
plot_validation(y_test, y_pred_rgs, 'Random Forest')
plot_validation(y_test, y_pred_svr, 'SVR')
plot_validation(y_test, y_pred_lgr, 'Linear Regression')
