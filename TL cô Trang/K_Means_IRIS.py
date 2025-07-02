from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
import numpy as np

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data             # Dữ liệu đặc trưng (bỏ nhãn loài)
y_true = iris.target      # Nhãn thật để đánh giá
target_names = iris.target_names  # Tên các loài hoa

# Khởi tạo và huấn luyện mô hình K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X)

# Gắn nhãn cụm cho từng điểm dữ liệu
labels = kmeans.labels_

# Hàm ánh xạ cụm → nhãn thật dựa trên số lượng chiếm ưu thế
def map_clusters_to_labels(y_true, y_pred):
    mapped = np.zeros_like(y_pred)
    for i in range(3):
        mask = (y_pred == i)
        mapped[mask] = mode(y_true[mask])[0]
    return mapped

# Ánh xạ cụm → tên loài hoa
cluster_to_species = {}
for i in range(3):
    #cluster_to_species[i] = target_names[mode(y_true[labels == i])[0][0]]
    cluster_to_species[i] = target_names[mode(y_true[labels == i], keepdims=True).mode[0]]

# Nhập dữ liệu hoa từ bàn phím
print("\nNhập 4 đặc trưng (sepal length, sepal width, petal length, petal width), cách nhau bởi dấu cách:")
input_data = input().strip()
try:
    flower = np.array([float(x) for x in input_data.split()]).reshape(1, -1)
    prediction = kmeans.predict(flower)[0]
    predicted_species = cluster_to_species[prediction]
    print(f"Dự đoán cụm: {prediction}")
    print(f"Dự đoán loài hoa gần đúng: {predicted_species}")
except:
    print("Dữ liệu không hợp lệ. Hãy nhập 4 giá trị số cách nhau bằng dấu cách.")

# Đánh giá độ chính xác
mapped_labels = map_clusters_to_labels(y_true, labels)
accuracy = accuracy_score(y_true, mapped_labels)
print(f"\n Độ chính xác phân cụm (sau ánh xạ): {accuracy:.2f}")
