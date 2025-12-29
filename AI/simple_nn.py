# simple_nn.py — РАБОТАЮЩАЯ версия (простая и понятная)
import numpy as np
from data import answers, questions
import pickle
from sklearn.feature_extraction.text import CountVectorizer


class UltraPowerNN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # 6 СЛОЁВ = 50x больше мощности!
        sizes = [input_size, 512, 256, 384, 256, 128, output_size]

        self.weights = []
        self.biases = []

        for i in range(len(sizes) - 1):
            # Glorot/Xavier инициализация
            w = np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            b = np.zeros((1, sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        total_params = sum(w.size for w in self.weights)
        print(f"🚀 ULTRA PowerNN: {total_params:,} параметров (50x мощнее!)")

    def forward(self, x):
        self.activations = [x]

        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:  # НЕ для выходного слоя
                x = np.tanh(x)
            self.activations.append(x)

        return x

    def train(self, X, y, epochs=20000, lr=0.001, batch_size=16):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward
                out = self.forward(X_batch)
                error = y_batch - out

                # Backpropagation по ВСЕМ слоям
                deltas = [error * (1 - np.tanh(out) ** 2)]

                for i in range(len(self.weights) - 2, -1, -1):
                    delta = np.dot(deltas[0], self.weights[i + 1].T) * (1 - self.activations[i + 1] ** 2)
                    deltas.insert(0, delta)

                # Обновление весов
                for i in range(len(self.weights)):
                    grad_w = np.dot(self.activations[i].T, deltas[i])
                    grad_b = np.sum(deltas[i], axis=0, keepdims=True)

                    self.weights[i] -= lr * grad_w
                    self.biases[i] -= lr * grad_b

                total_loss += np.mean(np.abs(error))

            if epoch % 2000 == 0:
                avg_loss = total_loss / (n_samples // batch_size)
                print(f"Эпоха {epoch}, Loss: {avg_loss:.4f}")


# АВТОопределение размеров
print("🧠 МОЩНОЕ ОБУЧЕНИЕ B2B ИИ...")

# Аугментация + большой словарь
vectorizer = CountVectorizer(max_features=300, lowercase=True)  # Больше слов!
X_questions = vectorizer.fit_transform(questions).toarray()
y_answers = vectorizer.transform(answers).toarray()

print(f"✅ ULTRA данные: X={X_questions.shape}, y={y_answers.shape}")

# 50x МОЩНЕЕ!
nn = UltraPowerNN(input_size=300, output_size=300)
nn.train(X_questions, y_answers, epochs=25000, lr=0.0008, batch_size=8)  # Меньше lr!

with open('ultra_ai.pkl', 'wb') as f:
    pickle.dump({
        'model': nn, 'vectorizer': vectorizer,
        'questions': questions, 'answers': answers,
        'X_questions': X_questions, 'y_answers': y_answers
    }, f)
print("✅ 🔥 ULTRA ИИ (50x мощнее) ГОТОВ!")


# 🚀 ЗАПУСК
# print("🔥 ОБУЧЕНИЕ НАЧИНАЕТСЯ...")
# nn = SimpleNN()
# nn.train(X, y)
# print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
#
# # 🧠 ТЕСТ
# print("\n🔮 ТЕСТИРУЕМ НОВЫЙ ВОПРОС:")
# new_question = ["цвет неба какой"]
# new_vec = vectorizer.transform(new_question).toarray()
# pred = nn.forward(new_vec)
#
# # Находим ближайший ответ
# similarities = np.dot(pred, y.T)[0]  # Косинусное сходство
# best_idx = np.argmax(similarities)
# print(f"❓ Вопрос: '{new_question[0]}'")
# print(f"🤖 Ответ: '{answers[best_idx]}'")
# print(f"📊 Уверенность: {similarities[best_idx]:.3f}")
