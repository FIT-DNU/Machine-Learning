import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bộ dữ liệu điểm thi
mid_scores = np.array([2, 3, 3.5, 5, 6, 6.5, 7.5, 8, 8.5, 9, 9.5, 10])
final_scores = np.array([1, 2, 3, 4, 6, 6, 7, 8, 9, 9.5, 10, 10])

# Nhãn: Đạt (1) hoặc Không đạt (0)
labels = np.where((mid_scores + final_scores) / 2 >= 5, 1, 0)

# Chuyển thành DataFrame
df = pd.DataFrame({'Midterm': mid_scores, 'Final': final_scores, 'Pass': labels})

# Trực quan hóa dữ liệu
plt.scatter(df['Midterm'], df['Final'], c=df['Pass'], cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Midterm Score')
plt.ylabel('Final Score')
plt.title('Student Classification with KNN')
plt.show()

# Chuẩn bị dữ liệu cho KNN
X = df[['Midterm', 'Final']].values
y = df['Pass'].values

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Áp dụng KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Hiển thị kết quả
print(f"Độ chính xác của mô hình KNN: {accuracy:.2f}")
print(f"Dự đoán nhãn cho tập kiểm tra: {y_pred}")
