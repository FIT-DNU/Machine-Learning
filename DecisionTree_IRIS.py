import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Tạo dữ liệu mô phỏng
data = {
    'Age': [25, 30, 45, 35, 22, 60, 48, 33],
    'HighIncome': [1, 0, 1, 0, 0, 1, 1, 0],
    'Student': [1, 1, 0, 0, 1, 0, 0, 1],
    'BuyComputer': [1, 1, 1, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df[['Age', 'HighIncome', 'Student']]
y = df['BuyComputer']

# Huấn luyện cây
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X, y)

# Vẽ cây
plt.figure(figsize=(8,5))
plot_tree(tree, feature_names=X.columns, class_names=["Không mua", "Mua"], filled=True)
plt.show()
