import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Tải dữ liệu từ URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Chia dữ liệu
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)

# Dự đoán
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Vẽ cây
plt.figure(figsize=(16,8))
plot_tree(tree, feature_names=X.columns, class_names=["Không", "Có"], filled=True)
plt.show()

print(f"Độ chính xác: {accuracy:.2f}")
