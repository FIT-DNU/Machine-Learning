import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Tạo dữ liệu giả lập
data = {
    'Age': [25, 45, 35, 33, 52, 23, 40, 60, 48, 36],
    'Income': [50000, 64000, 58000, 52000, 83000, 48000, 61000, 90000, 72000, 65000],
    'MaritalStatus': ['Single', 'Married', 'Single', 'Single', 'Married', 'Single', 'Married', 'Married', 'Single', 'Married'],
    'CreditScore': [650, 720, 690, 640, 710, 670, 700, 730, 710, 680],
    'Responded': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Mã hóa nhãn và đặc trưng phân loại
le_marital = LabelEncoder()
df['MaritalStatus'] = le_marital.fit_transform(df['MaritalStatus'])
le_resp = LabelEncoder()
df['Responded'] = le_resp.fit_transform(df['Responded'])

X = df.drop('Responded', axis=1)
y = df['Responded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
