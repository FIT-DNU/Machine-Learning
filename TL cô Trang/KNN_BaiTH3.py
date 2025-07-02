import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bộ dữ liệu BMI và mức đường huyết
bmi = np.array([18, 21, 22, 25, 27, 29, 30, 31, 33, 35, 38, 40])
glucose = np.array([90, 100, 110, 115, 120, 125, 130, 135, 140, 145, 150, 160])

# Gắn nhãn bệnh tiểu đường
labels = np.where((glucose >= 126) | (bmi >= 30), 1, 0)

# Chuyển thành DataFrame
df = pd.DataFrame({'BMI': bmi, 'Glucose': glucose, 'Diabetes': labels})

# Trực quan hóa dữ liệu
plt.scatter(df['BMI'], df['Glucose'], c=df['Diabetes'], cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('BMI')
plt.ylabel('Glucose Level')
plt.title('Diabetes Classification with KNN')
plt.colorbar(label='Diabetes (1 = Yes, 0 = No)')
plt.show()

# Chuẩn bị dữ liệu
X = df[['BMI', 'Glucose']].values
y = df['Diabetes'].values

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Hiển thị kết quả
print(f"Độ chính xác của mô hình KNN: {accuracy:.2f}")
print(f"Dự đoán bệnh tiểu đường: {y_pred}")
