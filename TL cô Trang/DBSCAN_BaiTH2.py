import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Tạo DataFrame từ dữ liệu khách hàng
data = {
    'Khách hàng': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Độ tuổi': [35, 40, 30, 25, 45, 50, 60, 65, 60, 70],
    'Mức chi tiêu': [1800, 1200, 1000, 1500, 2000, 1000, 800, 900, 850, 700]
}

df = pd.DataFrame(data)

# Chọn các thuộc tính để phân cụm
X = df[['Độ tuổi', 'Mức chi tiêu']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sử dụng DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2)
labels = dbscan.fit_predict(X_scaled)

# Thêm nhãn cụm vào DataFrame
df['Nhãn cụm'] = labels

# In DataFrame kết quả
print(df)

# Biểu diễn kết quả phân cụm
plt.figure(figsize=(8, 6))
plt.scatter(df['Độ tuổi'], df['Mức chi tiêu'], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel('Độ tuổi')
plt.ylabel('Mức chi tiêu')
plt.title('DBSCAN Clustering Result')
plt.show()
