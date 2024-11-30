import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

# Carregar o dataset
df_train = pd.read_csv('train.csv')

# Contar o número total de entradas NaN no df_train
total_nan = df_train.isna().sum().sum()
print(f"O DataFrame df_train contém {total_nan} entradas NaN.")

# Separar target (variável dependente)
y_train = df_train['SalePrice']

# Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
X_train = df_train.drop(['Id', 'SalePrice'], axis=1)

# Separar colunas numéricas e categóricas
X_train_numeric = X_train.select_dtypes(include=['number'])
X_train_categorical = X_train.select_dtypes(include=['object'])

# Usar SimpleImputer para preencher valores nulos nas colunas numéricas
numeric_imputer = SimpleImputer(strategy='mean')
X_train_numeric = pd.DataFrame(numeric_imputer.fit_transform(X_train_numeric), columns=X_train_numeric.columns)

# Usar SimpleImputer para preencher valores nulos nas colunas categóricas
categorical_imputer = SimpleImputer(strategy='constant', fill_value='None')
X_train_categorical = pd.DataFrame(categorical_imputer.fit_transform(X_train_categorical), columns=X_train_categorical.columns)

# Combinar as colunas numéricas e categóricas novamente
X_train = pd.concat([X_train_numeric, X_train_categorical], axis=1)

# Codificar variáveis categóricas
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])

# Definir o modelo Random Forest
rf = RandomForestRegressor(random_state=42)

# Definir os hiperparâmetros a serem testados
param_grid = {
    'n_estimators': [50, 100, 200],  # Número de árvores
    'max_features': [1.0, 'sqrt'],  # Número de features a serem consideradas para a melhor divisão
    'max_depth': [None, 10, 20, 30],  # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras necessárias para dividir um nó
}

# Implementar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Exibir os melhores parâmetros encontrados
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_}")

# Ajustar o modelo com os melhores parâmetros
best_rf = grid_search.best_estimator_

# Verificar se há valores NaN ou Inf nos dados de treinamento
if X_train.isnull().values.any() or np.isinf(X_train).values.any():
    print("Valores NaN ou Inf encontrados em X_train. Corrija antes de prosseguir.")
else:
    print("X_train está limpo.")

# # Definir os hiperparâmetros a serem testados
# param_dist = {
#     'n_estimators': [10, 50, 100, 200, 500],
#     'max_features': [1.0, 'sqrt', 'log2', 0.5, 0.75],
#     'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40],
#     'min_samples_split': [2, 5, 10, 15, 20],
#     'min_samples_leaf': [1, 2, 5, 10],
#     'bootstrap': [True, False],
#     'max_samples': [0.5, 0.75, 1.0],
#     'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
#     'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
#     'criterion': ['squared_error', 'absolute_error'],
# }

# # Implementar RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
#                                    n_iter=200, cv=5, n_jobs=-1, verbose=2, random_state=42)
# random_search.fit(X_train, y_train)

# # Exibir os melhores parâmetros encontrados
# print(f"Melhores parâmetros: {random_search.best_params_}")
# print(f"Melhor score: {random_search.best_score_}")

# # Ajustar o modelo com os melhores parâmetros
# best_rf = random_search.best_estimator_

# Carregar o dataset de teste
df_test = pd.read_csv('test.csv')

# Remover colunas desnecessárias, incluindo o ID
X_test = df_test.drop(['Id'], axis=1)
ids = df_test['Id']

# Separar colunas numéricas e categóricas no conjunto de teste
X_test_numeric = X_test.select_dtypes(include=['number'])
X_test_categorical = X_test.select_dtypes(include=['object'])

# Usar SimpleImputer para preencher valores nulos nas colunas numéricas
X_test_numeric = pd.DataFrame(numeric_imputer.transform(X_test_numeric), columns=X_test_numeric.columns)

# Usar SimpleImputer para preencher valores nulos nas colunas categóricas
X_test_categorical = pd.DataFrame(categorical_imputer.transform(X_test_categorical), columns=X_test_categorical.columns)

# Combinar as colunas numéricas e categóricas novamente
X_test = pd.concat([X_test_numeric, X_test_categorical], axis=1)

# Codificar variáveis categóricas
for column in X_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_test[column] = le.fit_transform(X_test[column])

# Fazer previsões
y_pred = best_rf.predict(X_test)

# Abrir um arquivo para escrita no formato CSV
with open('predictions.csv', 'w') as f:
    # Escrever o cabeçalho
    f.write("Id,SalePrice\n")
    
    # Escrever as previsões
    for id, saleprice in zip(ids, y_pred):
        f.write(f"{id},{saleprice}\n")

# Contar o número de linhas no arquivo CSV
with open('predictions.csv', 'r') as f:
    line_count = sum(1 for line in f)  # Conta cada linha no arquivo

print(f"O arquivo predictions.csv tem {line_count} linhas.")