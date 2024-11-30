import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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

# Treinar o modelo Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Fazer previsões
y_pred = gb.predict(X_test)

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