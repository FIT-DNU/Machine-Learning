import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bộ dữ liệu điểm chuyên cần và thái độ
attendance = np.array([2, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9, 9.5, 10])
behavior = np.array([3, 4, 5, 6, 7, 7, 7.5, 8, 9, 9.5, 9.5, 10])

# Tính điểm rèn luyện
train_score = 0.6 * attendance + 0.4 * behavior

# Gắn nhãn rèn luyện
labels = np.where(train_score < 5, 0, np.where(train_score < 8, 1, 2))

# Chuyển thành DataFrame
df = pd.DataFrame({'Attendance': attendance, 'Behavior': behavior, 'Training Score': train_score, 'Label': labels})

# Trực quan hóa dữ liệu
plt.scatter(df['Attendance'], df['Behavior'], c=df['Label'], cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Attendance Score')
plt.ylabel('Behavior Score')
plt.title('Student Training Score Classification')
plt.colorbar(label='Training Level')
plt.show()

# Chuẩn bị dữ liệu
X = df[['Attendance', 'Behavior']].values
y = df['Label'].values

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Hiển thị kết quả
print(f"Độ chính xác của mô hình KNN: {accuracy:.2f}")
print(f"Dự đoán nhãn rèn luyện: {y_pred}")
