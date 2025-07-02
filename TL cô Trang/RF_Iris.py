from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Dữ liệu
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Kết quả
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
importances = model.feature_importances_
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Tầm quan trọng
