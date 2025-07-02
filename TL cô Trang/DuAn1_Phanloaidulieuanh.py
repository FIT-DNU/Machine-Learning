import os
import pickle

# Thư viện os được sử dụng để truy cập vào các thư mục và file trên hệ thống.
# Thư viện pickle được sử dụng để lưu trữ và truy xuất dữ liệu dưới dạng nhị phân.

import skimage
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Thư viện skimage.io được sử dụng để đọc ảnh từ file.
# Thư viện skimage.transform được sử dụng để thay đổi kích thước ảnh.
# Thư viện numpy được sử dụng để xử lý các mảng dữ liệu số.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Thư viện skimage.io được sử dụng để đọc ảnh từ file.
# Thư viện skimage.transform được sử dụng để thay đổi kích thước ảnh.
# Thư viện numpy được sử dụng để xử lý các mảng dữ liệu số.
# Thư viện sklearn.model_selection được sử dụng để chia dữ liệu thành tập huấn luyện và tập kiểm tra.
# Thư viện sklearn.model_selection được sử dụng để tìm các tham số tối ưu cho một mô hình học máy.
# Thư viện sklearn.svm được sử dụng để xây dựng mô hình phân loại bằng thuật toán SVM.
# Thư viện sklearn.metrics được sử dụng để đánh giá độ chính xác của một mô hình học máy.
# Chuẩn bị dữ liệu
input_dir = 'C:/Users/admin/Downloads/clf-data/clf-data'
categories = ['empty', 'not_empty']
data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)
# Lập qua một danh sách và in ra chỉ số và phần tử của danh sách
For index, element in enumerate([1,2,3,4,5]):
    print(index, element)
data = np.asarray(data)
labels = np.asarray(labels)
# train / test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(X_train, y_train)
# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print(f'Độ chính xác của mô hình SVM kết hợp với gridsearch là:{score * 100}')
