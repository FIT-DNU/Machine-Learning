import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sales_data = pd.read_csv('train.csv', index_col = 0)
sales_data.head(10)
sales_data.info()
monthly_data = sales_data.copy()
monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])
monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
monthly_data.date = pd.to_datetime(monthly_data.date)
monthly_data.info()
fig, ax = plt.subplots(figsize=(15,5))
# Biểu đồ số sản phẩm bán ra theo tháng
sns.lineplot(x='date', y = 'sales', data=monthly_data, ax=ax, color='mediumblue', label='Total Sales')

ax.set(xlabel = "Date",
       ylabel = "Sales",
       title = 'Số lượng sản phẩm được bán ra theo thời gian')

sns.despine()
# Thêm sales_diff để tính độ chênh lệch của các tháng
monthly_data['sales_diff'] = monthly_data.sales.diff()

# Xóa giá trị NA
monthly_data = monthly_data.dropna()
monthly_data.head(10)
supervised_df = monthly_data.copy()

# Xóa cột date và cột sales ra khỏi dataframe
# supervised_df = supervised_df.drop(['date','sales'], axis = 1)

# Tạo bản sao của monthly_data
for i in range(1,15):
    col_name = 'lag_' + str(i)
    supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

# Xóa những cột giá trị NA
supervised_df = supervised_df.dropna().reset_index(drop=True)

# Gắn date cho index và chuyển sang dữ liệu datetime
supervised_df = supervised_df.set_index('date')
supervised_df.index = pd.to_datetime(supervised_df.index)
supervised_df.head10)
# Lấy dữ liệu trừ 12 dòng cuối, và bỏ cột sales
train_data = supervised_df.iloc[:-12, 1:]

# Lấy dữ liệu 12 dòng cuối
test_data = supervised_df.iloc[-12:, 1:]

print("Train Data Shape :",train_data.shape)
print("Test Data Shape :",test_data.shape)
# Dùng để normalization(chuẩn hóa dữ liệu)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
train_data
test_data
X_train,y_train = train_data[:,1:],train_data[:,0:1]
X_test,y_test = test_data[:,1:],test_data[:,0:1]

# Biến đổi mảng 2D thành mảng 1D
y_train = y_train.ravel()
y_test = y_test.ravel()

print("X_train Shape :",X_train.shape)
print("y_train Shape :",y_train.shape)
print("X_test Shape :",X_test.shape)
print("X_test Shape :",y_test.shape)
# Gọi model LinearRegression
from sklearn.linear_model import LinearRegression
# Gọi model RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Gọi model Support vector machine regressor (SVR)
from sklearn.svm import SVR
# Các metrics dùng để đánh giá mô hình
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
# Xây dựng mô hình Random ForestRegressor
Rgs = RandomForestRegressor(n_estimators= 100).fit(X_train, y_train)
y_pred_rgs = Rgs.predict(X_test)

# Xây dựng mô hình SVR
svr = SVR().fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# Xây dựng mô hình Linear Regression
lgr = LinearRegression().fit(X_train, y_train)
y_pred_lgr = lgr.predict(X_test)
print(y_pred_rgs)
print(y_pred_svr)
print(y_pred_lgr.ravel())


def pre_test_set(y_pred):
    y_pred = y_pred.reshape(-1, 1)
    # Đây là Set Matrix và nó chứa các tính năng đầu vào của dữ liệu thử nghiệm cũng như đầu ra được dự đoán
    pre_test_set = np.concatenate([y_pred, X_test], axis=1)
    # Gắn giá trị y_pred rồi đến X_test, nên y_pred ở vị trí đầu

    # Convert scaler trở lại giá trị đầu
    pre_test_set = scaler.inverse_transform(pre_test_set)
    return pre_test_set


# Tạo dataframe chứa giá trị dự đoán (RF) cùng tập x (biến độc lập)
pre_test_set_rfs = pre_test_set(y_pred_rgs)

# Tạo dataframe chứa giá trị dự đoán (SVR) cùng tập x (biến độc lập)
pre_test_set_svr = pre_test_set(y_pred_svr)

# Tạo dataframe chứa giá trị dự đoán (Linear Regression) cùng tập x (biến độc lập)
pre_test_set_lgr = pre_test_set(y_pred_lgr)
# Tạo một dataframe chứa cột ngày tháng của tập dữ đoán (12 tháng cuối)
sales_dates = monthly_data['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
# Số lượng sản phẩm thực tế của 12 tháng cuối
actual_sales = monthly_data['sales'][-13:].to_list()
pre_test_set_rfs
# Tạo một dataframe chứa cột ngày tháng của tập dữ đoán (12 tháng cuối)
sales_dates = monthly_data['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

# Số lượng sản phẩm thực tế của 12 tháng cuối
actual_sales = monthly_data['sales'][-13:].to_list()
def predict_data(pre_test_set):
    result_list = []
    for index in range(0,len(pre_test_set)):
        result_list.append(pre_test_set[index][0] + actual_sales[index])

    # Tạo series cho giá trị số lượng dự đoán
    pre_series =  pd.Series(result_list,name="Predictions")
    predict = predict_df.merge(pre_series,left_index=True,right_index=True)
    return predict
# Tạo cho mô hình randomforest
predict_df_RF = predict_data(pre_test_set_rfs)

# Tạo cho mô hình SVR
predict_df_SVR = predict_data(pre_test_set_svr)

# Tạo cho mô hình Linear Regression
predict_df_lgr = predict_data(pre_test_set_lgr)
predict_df_RF
# Tạo cho mô hình randomforest
predict_df_RF = predict_data(pre_test_set_rfs)

# Tạo cho mô hình SVR
predict_df_SVR = predict_data(pre_test_set_svr)

# Tạo cho mô hình Linear Regression
predict_df_lgr = predict_data(pre_test_set_lgr)
predict_df_RF
# Tạo cho mô hình randomforest
predict_df_RF = predict_data(pre_test_set_rfs)

# Tạo cho mô hình SVR
predict_df_SVR = predict_data(pre_test_set_svr)

# Tạo cho mô hình Linear Regression
predict_df_lgr = predict_data(pre_test_set_lgr)
predict_df_RF
# kết quả đánh giá của module RandomForest
print_evaluate(predict_df_RF, 'Random Forest')

# kết quả đánh giá của module SVR
print_evaluate(predict_df_SVR, 'SVR')

# kết quả đánh giá của module Linear Regression
print_evaluate(predict_df_lgr, 'Linear Regression')
plot_results(predict_df_RF, 'Random Forest')
plot_results(predict_df_SVR, 'SVR')
plot_results( predict_df_lgr, 'Linear Regression')
