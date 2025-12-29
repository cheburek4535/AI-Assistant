from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data import questions, answers

# ОБЪЕДИНЯЕМ ВОПРОСЫ+ОТВЕТЫ в один список → ЕДИНЫЙ словарь 20 слов
all_texts = questions + answers  # ["какого...", "небо...", "что...", "python..."]


# Создаём векторизатор: max_features=20 = ровно 20 чисел на вход/выход
vectorizer = CountVectorizer(max_features=100, lowercase=True)
X = vectorizer.fit_transform(questions).toarray()  # (50,100)
y = X.copy()  # Autoencoder: учим восстанавливать вход

# print("✅ Фиксированные размеры: X.shape =", X.shape, "y.shape =", y.shape)
# print("Первые 3 слова из словаря:", vectorizer.get_feature_names_out()[:3])
# print("Пример вектора вопроса 0:", X[0][:5])  # Первые 5 чисел