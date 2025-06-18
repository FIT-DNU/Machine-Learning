import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Bước 1: Tạo dữ liệu huấn luyện
data = pd.DataFrame({
    'MauSac': ['Do', 'Vang', 'Do', 'Xanh', 'Vang'],
    'KichThuoc': ['Nho', 'Lon', 'Lon', 'Nho', 'Nho'],
    'Loai': ['Tao', 'Chuoi', 'Tao', 'Tao', 'Chuoi']
})

# Bước 2: Mã hóa dữ liệu phân loại thành số để mô hình có thể xử lý
le_color = LabelEncoder()
le_size = LabelEncoder()
le_label = LabelEncoder()
X = pd.DataFrame({
    'MauSac': le_color.fit_transform(data['MauSac']),
    'KichThuoc': le_size.fit_transform(data['KichThuoc'])
})
y = le_label.fit_transform(data['Loai'])

# Bước 3: Huấn luyện mô hình Naïve Bayes với dữ liệu đã mã hóa
model = CategoricalNB()
model.fit(X, y)

# Bước 4: Dự đoán mẫu mới (Vang, Lon)
new_sample = pd.DataFrame({
    'MauSac': [le_color.transform(['Vang'])[0]],
    'KichThuoc': [le_size.transform(['Lon'])[0]]
})

# Bước 5: Dự đoán và in ra kết quả theo tên gốc
pred = model.predict(new_sample)
print("Kết quả dự đoán:", le_label.inverse_transform(pred))
