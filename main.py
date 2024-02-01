import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Загрузка данных
df = pd.read_csv('parkinsons.data')
print(df.head())

# Выделение признаков и меток
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Нормализация признаков
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2, train_size=0.8)

# Инициализация XGBoost классификатора
model = XGBClassifier()

# Гиперпараметрическая настройка с помощью Grid Search
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_

# Обучение модели на обучающих данных
best_model.fit(X_train, Y_train)

# Прогнозирование на тестовых данных и оценка точности
Y_hat = best_model.predict(X_test)
print(accuracy_score(Y_test, Y_hat))