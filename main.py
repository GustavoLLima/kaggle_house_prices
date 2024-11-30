import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np  # Importar numpy para calcular a raiz quadrada

def preprocess(X):
    # Pré-processamento de dados, preenchimento de valores nulos e codificação de variáveis categóricas
    X.fillna(X.mean(numeric_only=True), inplace=True)  # Preenche NaN em colunas numéricas com a média
    X.fillna('None', inplace=True)  # Preenche NaN em colunas categóricas com um valor padrão

    # Codificar variáveis categóricas
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    return X

def preprocess_train_dataset():
    # Carregar o dataset
    df = pd.read_csv("train.csv")

    # Separar target (variável dependente)
    y = df['SalePrice']

    # Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
    X = df.drop(['Id', 'SalePrice'], axis=1)

    X = preprocess(X)
    return X, y

def preprocess_test_dataset():
    # Carregar o dataset
    df_test = pd.read_csv('test.csv')

    # Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
    X = df_test.drop(['Id'], axis=1)
    ids = df_test['Id']

    X = preprocess(X)
    return X, ids

# Pré-processar os dados
X_train, y_train = preprocess_train_dataset()
X_test, ids = preprocess_test_dataset()

# Definir os modelos
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel='linear')  # Você pode ajustar o kernel conforme necessário
}

# Avaliar cada modelo com validação cruzada
results = {}

for model_name, model in models.items():
    # Realizar validação cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Calcular MSE médio e RMSE
    mse = -cv_scores.mean()  # O cross_val_score retorna valores negativos para MSE
    rmse = np.sqrt(mse)  # Calcular RMSE
    mae = mean_absolute_error(y_train, model.fit(X_train, y_train).predict(X_train))  # MAE no conjunto de treinamento
    r2 = r2_score(y_train, model.fit(X_train, y_train).predict(X_train))  # R² no conjunto de treinamento

    # Armazenar os resultados
    results[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,  # Adicionar RMSE aos resultados
        "R²": r2
    }

# Exibir os resultados
print("Desempenho dos modelos com validação cruzada:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Erro Médio Absoluto (MAE): {metrics['MAE']}")
    print(f"  Erro Quadrático Médio (MSE): {metrics['MSE']}")
    print(f"  Raiz do Erro Quadrático Médio (RMSE): {metrics['RMSE']}")
    print(f"  Coeficiente de Determinação (R²): {metrics['R²']}")
    print()

# Encontrar o modelo com o menor RMSE
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = models[best_model_name]

print(f"O melhor modelo com base no RMSE é: {best_model_name}")

# Fazer previsões no conjunto de teste com o melhor modelo
best_model.fit(X_train, y_train)  # Treinar o melhor modelo com todos os dados de treinamento
y_pred = best_model.predict(X_test)

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