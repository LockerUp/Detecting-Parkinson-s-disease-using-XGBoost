import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# загрузка датасета
df = pd.read_csv('parkinsons.data')

# извлечение объектов и меток
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# нормализатор
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)

# Разделите набор данных на обучающий(80%) и тестовый(20%) наборы.
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, train_size=0.8)

# Инициализация классификатора XGBoost
model = XGBClassifier()
# Обучаем модель на обучающих данных
model.fit(X_train, Y_train)

# Делаем прогнозы и оцениваем точность
Y_hat = [round(yhat) for yhat in model.predict(X_test)]
print(accuracy_score(Y_test, Y_hat))

