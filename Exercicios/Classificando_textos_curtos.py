
from sklearn.feature_extraction.text import CountVectorizer # Importa a classe CountVectorizer para converter texto em uma matriz de contagens de tokens
from sklearn.model_selection import train_test_split # Importa a função train_test_split para dividir dados em conjuntos de treinamento e teste
from sklearn.naive_bayes import MultinomialNB # Importa a classe MultinomialNB, um classificador Naive Bayes adequado para contagens de texto
from sklearn.metrics import accuracy_score # Importa a função accuracy_score para avaliar a precisão do modelo

# Dados de exemplo
textos = [ # Cria uma lista de strings representando documentos de texto
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"] # Cria uma lista de strings representando as categorias correspondentes aos textos

# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer() # Cria uma instância da classe CountVectorizer
X = vectorizer.fit_transform(textos) # Aprende o vocabulário dos textos e transforma os textos em uma matriz de contagens de tokens

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42) # Divide os dados em 50% para treinamento e 50% para teste, com um estado aleatório fixo para reprodutibilidade

# Treinando o classificador
clf = MultinomialNB() # Cria uma instância do classificador MultinomialNB
clf.fit(X_train, y_train) # Treina o classificador usando os dados de treinamento (textos e categorias)

# Predição e Avaliação
y_pred = clf.predict(X_test) # Faz previsões nas categorias dos textos do conjunto de teste usando o modelo treinado
print(f"Acurácia: {accuracy_score(y_test, y_pred)}") # Calcula e imprime a acurácia do modelo comparando as previsões com as categorias verdadeiras do conjunto de teste
