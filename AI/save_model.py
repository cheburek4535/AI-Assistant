# save_model.py — СОХРАНЯЕМ МОЗГ нейросети
# import pickle
# from simple_nn import SimpleNN
# from vectorize import X, y, vectorizer
# from data import answers
#
# if __name__ == '__main__':
#     print("🧠 Обучаем под новый датасет...")
#
#     # АВТОопределение размеров по vectorizer
#     vocab_size = len(vectorizer.get_feature_names_out())
#     print(f"📊 Словарь: {vocab_size} слов")
#
#     nn = SimpleNN(
#         input_size=vocab_size,
#         hidden_size=max(50, vocab_size // 3),  # Адаптивно
#         output_size=vocab_size
#     )
#
#     nn.train(X, y, epochs=5000, lr=0.003)  # Больше эпох для большой сети
#
#     with open('first_ai.pkl', 'wb') as f:
#         pickle.dump({
#             'model': nn,
#             'vectorizer': vectorizer,
#             'answers': answers,
#             'y': y,
#             'vocab': vectorizer.get_feature_names_out()
#         }, f)
#
#
#     print("💾 МОЗГ СОХРАНЁН в first_ai.pkl!")