# # test_saved.py — ТЕСТИРУЕМ сохранённый МОЗГ
# import pickle
# import numpy as np
#
# # 1. ЗАГРУЖАЕМ обученную нейросеть
# with open('first_ai.pkl', 'rb') as f:
#     brain = pickle.load(f)
#     model = brain['model']
#     vectorizer = brain['vectorizer']
#     answers = brain['answers']
#
# print("🧠 МОЗГ ЗАГРУЖЁН! Тестируем...")
# print("Ключи в brain:", brain.keys())
# print("Размер answers:", len(answers))
#
#
# # 2. ТЕСТИРУЕМ несколько вопросов
# test_questions = [
#     "проблема с БД",
#     "а где документация",
#     "привет",  # ТОЧНО из датасета!
#     "что такое postgres"
# ]
#
# for question in test_questions:
#     # Векторизуем
#     vec = vectorizer.transform([question]).toarray()
#     pred = model.forward(vec)
#
#     # УМНЫЙ поиск: НЕ просто косинус, а ДИСТАНЦИЯ
#     distances = np.linalg.norm(pred - brain['y'], axis=1)  # Евклидова дистанция
#     best_idx = np.argmin(distances)  # БЛИЖАЙШИЙ ответ
#
#     print(f"\n❓ '{question}'")
#     print(f"🤖 '{answers[best_idx]}'")
#     print(f"📏 Дистанция: {distances[best_idx]:.3f} (меньше = лучше)")
