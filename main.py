import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np  # Importar numpy para calcular a raiz quadrada

def preprocess(X):
    print("Iniciando o pré-processamento dos dados...")
    # Pré-processamento de dados, preenchimento de valores nulos e codificação de variáveis categóricas
    X.fillna(X.mean(numeric_only=True), inplace=True)  # Preenche NaN em colunas numéricas com a média
    X.fillna('None', inplace=True)  # Preenche NaN em colunas categóricas com um valor padrão

    # Codificar variáveis categóricas
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    print("Pré-processamento concluído.")
    return X

def create_statistical_features(X):
    print("Criando features estatísticas...")
    # Criar novas features estatísticas
    X['mean'] = X.mean(axis=1)
    X['std'] = X.std(axis=1)
    X['min'] = X.min(axis=1)
    X['max'] = X.max(axis=1)
    X['median'] = X.median(axis=1)
    print("Features estatísticas criadas.")
    return X

def preprocess_train_dataset():
    print("Carregando o dataset de treinamento...")
    # Carregar o dataset
    df = pd.read_csv("train.csv")

    # Separar target (variável dependente)
    y = df['SalePrice']

    # Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
    X = df.drop(['Id', 'SalePrice'], axis=1)

    print("Dataset de treinamento carregado.")
    return X, y

def preprocess_test_dataset():
    print("Carregando o dataset de teste...")
    # Carregar o dataset
    df_test = pd.read_csv('test.csv')

    # Remover colunas desnecessárias, incluindo o ID e a coluna target do conjunto de features
    X = df_test.drop(['Id'], axis=1)
    ids = df_test['Id']

    print("Dataset de teste carregado.")
    return X, ids

def evaluate_models(X, y, models):
    print("Iniciando a avaliação dos modelos...")
    # Avaliar cada modelo com validação cruzada
    results = {}

    for model_name, model in models.items():
        print(f"Avaliando o modelo: {model_name}...")
        # Realizar validação cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        # Calcular MSE médio e RMSE
        mse = -cv_scores.mean()  # O cross_val_score retorna valores negativos para MSE
        rmse = np.sqrt(mse)  # Calcular RMSE
        mae = mean_absolute_error(y, model.fit(X, y).predict(X))  # MAE no conjunto de treinamento
        r2 = r2_score(y, model.fit(X, y).predict(X))  # R² no conjunto de treinamento

        # Armazenar os resultados
        results[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,  # Adicionar RMSE aos resultados
            "R²": r2
        }
        print(f"Modelo {model_name} avaliado com sucesso.")

    print("Avaliação dos modelos concluída.")
    return results

# Definir os modelos fora da função
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel='linear')  # Você pode ajustar o kernel conforme necessário
}

# Teste 1: Dados Crus
print("Iniciando teste com dados crus...")
X_train_raw, y_train = preprocess_train_dataset()
X_train_raw = preprocess(X_train_raw)  # Pré-processar dados crus
results_raw = evaluate_models(X_train_raw, y_train, models)

# Teste 2: Features Estatísticas
print("Iniciando teste com features estatísticas...")
X_train_stat, y_train = preprocess_train_dataset()
X_train_stat = preprocess(X_train_stat)  # Pré-processar
X_train_stat = create_statistical_features(X_train_stat)  # Adicionar features estatísticas
results_stat = evaluate_models(X_train_stat, y_train, models)

# Teste 3: Combinação de Dados Crus e Features Estatísticas
print("Iniciando teste com dados crus e features estatísticas...")
X_train_combined = preprocess(X_train_raw.copy())  # Pré-processar dados crus
X_train_combined = create_statistical_features(X_train_combined)  # Adicionar features estatísticas
results_combined = evaluate_models(X_train_combined, y_train, models)

# Exibir os resultados
print("Desempenho dos modelos com dados crus:")
for model_name, metrics in results_raw.items():
    print(f"{model_name}: {metrics}")
print()

print("Desempenho dos modelos com features estatísticas:")
for model_name, metrics in results_stat.items():
    print(f"{model_name}: {metrics}")
print()

print("Desempenho dos modelos com dados crus e features estatísticas:")
for model_name, metrics in results_combined.items():
    print(f"{model_name}: {metrics}")
print()

# Encontrar o melhor modelo entre todos os testes
best_model_name_raw = min(results_raw, key=lambda x: results_raw[x]['RMSE'])
best_model_name_stat = min(results_stat, key=lambda x: results_stat[x]['RMSE'])
best_model_name_combined = min(results_combined, key=lambda x: results_combined[x]['RMSE'])

# Comparar os melhores modelos
best_models = {
    "Raw": best_model_name_raw,
    "Estatísticas": best_model_name_stat,
    "Combinado": best_model_name_combined
}

# Encontrar o melhor modelo geral
best_overall_model = min(best_models, key=lambda x: results_raw[best_models[x]]['RMSE'] if x == "Raw" else
                          results_stat[best_models[x]]['RMSE'] if x == "Estatísticas" else
                          results_combined[best_models[x]]['RMSE'])

print(f"O melhor modelo geral é: {best_models[best_overall_model]} usando dados {best_overall_model}.")

# Fazer previsões no conjunto de teste com o melhor modelo
# Determinar qual modelo usar com base na melhor opção
if best_overall_model == "Raw":
    best_model = models[best_model_name_raw]
    X_train_best = X_train_raw
elif best_overall_model == "Estatísticas":
    best_model = models[best_model_name_stat]
    X_train_best = X_train_stat
else:
    best_model = models[best_model_name_combined]
    X_train_best = X_train_combined

# Treinar o melhor modelo com todos os dados de treinamento
print(f"Treinando o melhor modelo: {best_models[best_overall_model]}...")
best_model.fit(X_train_best, y_train)

# Pré-processar os dados de teste
X_test, ids = preprocess_test_dataset()
X_test = preprocess(X_test)  # Pré-processar dados de teste

# Adicionar features estatísticas se o melhor modelo for o combinado
if best_overall_model == "Combinado":
    X_test = create_statistical_features(X_test)

# Fazer previsões
print("Fazendo previsões no conjunto de teste...")
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