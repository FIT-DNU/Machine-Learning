import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Tạo ngẫu nhiên 500 điểm trên mặt phẳng Oxy
np.random.seed(42)
data = np.random.rand(500, 2) * 10  # Giả sử phạm vi từ 0 đến 10

# Sử dụng phương pháp liên kết hoàn chỉnh để xây dựng cây phân cấp
linkage_matrix = linkage(data, method='complete')

# Vẽ dendrogram
plt.figure(figsize=(15, 6))
dendrogram(linkage_matrix, p=4, truncate_mode='level')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Chọn ngưỡng cắt để phân thành 4 cụm
threshold = 15  # Thay đổi ngưỡng theo yêu cầu của bạn

# Phân cụm dữ liệu sử dụng ngưỡng cắt
clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

# Biểu diễn kết quả phân cụm trên mặt phẳng Oxy với màu sắc tương phản
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='tab10', edgecolors='k', alpha=0.7)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Hierarchical Clustering Result')
plt.show()
