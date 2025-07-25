from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo
textos_originais = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional"
]
categorias_originais = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]

# Combine os textos e categorias originais para facilitar a impressão
dados_combinados = list(zip(textos_originais, categorias_originais))

# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([texto for texto, categoria in dados_combinados])

# Dividindo os dados em conjuntos de treinamento e teste
indices = list(range(len(dados_combinados)))
train_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=42)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = [categorias_originais[i] for i in train_indices]
y_test = [categorias_originais[i] for i in test_indices]
textos_train = [textos_originais[i] for i in train_indices]
textos_test = [textos_originais[i] for i in test_indices]

print("\nConjunto de Treinamento:")
for texto, caracteristicas, categoria in zip(textos_train, X_train.toarray(), y_train):
    print(f"Texto: {texto}")
    print(f"Características: {caracteristicas}")
    print(f"Categoria: {categoria}")

print("\nConjunto de Teste:")
for texto, caracteristicas, categoria in zip(textos_test, X_test.toarray(), y_test):
    print(f"Texto: {texto}")
    print(f"Características: {caracteristicas}")
    print(f"Categoria: {categoria}")

# Treinando o classificador
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predição e Avaliação
y_pred = clf.predict(X_test)
print(f"\nAcurácia: {accuracy_score(y_test, y_pred)}")