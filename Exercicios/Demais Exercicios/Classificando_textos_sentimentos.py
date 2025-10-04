from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo
textos_originais = [
    "Eu adorei o novo filme! As atuações foram incríveis.",
    "Este produto é horrível e não funciona como esperado.",
    "A comida estava deliciosa e o ambiente muito agradável.",
    "Fiquei extremamente decepcionado com o serviço.",
    "Que dia maravilhoso, cheio de sol e alegria!",
    "Hoje me senti muito triste e desanimado.",
    "O livro é muito bom, a história me prendeu do início ao fim.",
    "Não gostei nada daquele restaurante, a comida era sem graça.",
    "Recebi ótimas notícias hoje, estou muito feliz!",
    "Infelizmente, o evento foi cancelado.",
    "A palestra foi muito interessante e informativa.",
    "O atendimento ao cliente foi péssimo, não resolveram meu problema."
]
categorias_originais = ["positivo", "negativo", "positivo", "negativo", "positivo", "negativo", "positivo", "negativo", "positivo", "negativo", "positivo"]
# Combine os textos e categorias originais para facilitar a impressão
dados_combinados = list(zip(textos_originais, categorias_originais))

# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([texto for texto, categoria in dados_combinados])

# Imprimindo o vocabulário numerado
print("\n" + "="*30 + "\n")
print("Vocabulário do CountVectorizer:")
vocabulario = vectorizer.vocabulary_
vocabulario_numerado = sorted(vocabulario.items(), key=lambda item: item[1])
for indice, (palavra, _) in enumerate(vocabulario_numerado):
    print(f"{indice + 1}: {palavra}")
print("\n" + "="*30 + "\n")

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