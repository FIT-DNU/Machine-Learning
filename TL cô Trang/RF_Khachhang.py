import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dữ liệu mô phỏng
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 60, 200),
    'Income': np.random.randint(200, 1000, 200),
    'SpendingScore': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

# Gắn nhãn (giả sử): SpendingScore > 50 → VIP
df['Segment'] = (df['SpendingScore'] > 50).astype(int)

# Chia dữ liệu
X = df[['Age', 'Income', 'SpendingScore']]
y = df['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Kết quả
print("Độ chính xác:", accuracy_score(y_test, y_pred))
