import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
df_train = pd.read_csv('train.csv')

# Contar o número total de entradas NaN no df_train
total_nan = df_train.isna().sum().sum()

print(f"O DataFrame df_train contém {total_nan} entradas NaN.")

# Separar target (variável dependente)
y_train = df_train['SalePrice']

# Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
X_train = df_train.drop(['Id', 'SalePrice'], axis=1)

# Pré-processamento de dados, preenchimento de valores nulos e codificação de variáveis categóricas
X_train.fillna(X_train.mean(numeric_only=True), inplace=True)  # Preenche NaN em colunas numéricas com a média
X_train.fillna('None', inplace=True)  # Preenche NaN em colunas categóricas com um valor padrão

# Codificar variáveis categóricas
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])

# Carregar o dataset
df_test = pd.read_csv('test.csv')

# Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
X_test = df_test.drop(['Id'], axis=1)
ids = df_test['Id']

X_test.fillna(X_test.mean(numeric_only=True), inplace=True)  # Preenche NaN em colunas numéricas com a média
X_test.fillna('None', inplace=True)  # Preenche NaN em colunas categóricas com um valor padrão

# Codificar variáveis categóricas
for column in X_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_test[column] = le.fit_transform(X_test[column])

# Treinar o modelo Random Forest
rf = GradientBoostingRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Fazer previsões e calcular o erro médio quadrado
y_pred = rf.predict(X_test)

# for id, saleprice in zip(ids, y_pred):
#     print(f"Id: {id}, SalePrice: {saleprice}")

# Abrir um arquivo para escrita
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