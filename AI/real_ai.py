# real_ai.py — НАСТОЯЩИЙ ИИ: генерирует ответы СЛОВАМИ
import numpy as np
from data import questions
from simple_nn import UltraPowerNN
import pickle
import os

# Загружаем МОЩНЫЙ мозг
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'first_ai.pkl')

# Замените 'first_ai.pkl' на 'ultra_ai.pkl'
with open(os.path.join(script_dir, 'ultra_ai.pkl'), 'rb') as f:
    brain = pickle.load(f)
    nn = brain['model']
    vectorizer = brain['vectorizer']
    answers = brain['answers']
    y_answers = brain['y_answers']

class UltraAI:
    def __init__(self, brain):
        self.nn = brain['model']
        self.vectorizer = brain['vectorizer']
        self.answers = brain['answers']
        self.y_answers = brain['y_answers']

    def generate_answer(self, question):
        q_vec = self.vectorizer.transform([question]).toarray()
        pred_vec = self.nn.forward(q_vec)[0]
        distances = np.linalg.norm(pred_vec - self.y_answers, axis=1)
        best_idx = np.argmin(distances)
        return self.answers[best_idx], distances[best_idx], best_idx

ai = UltraAI(brain)

# ПОЛНЫЙ список тестов (45 вопросов в ТОЧНОМ порядке answers!)
tests = [
    "небо какого цвета бывает", "python это что за штука", "fastapi как стартануть",
    "бд не конектится почему", "доки где искать", "подключить постгрес к fastapi как",
    "ювикорн ошибка не пускает", "cors middleware fastapi где взять",
    "проверить jwt в хедере", "сессия sqlalchemy самозакрывается",
    "создать новую поставку как", "какие статусы у заказов есть",
    "скачать счёт-фактуру где", "стоимость доставки посчитать",
    "оплатить безналом инструкция", "алембик миграции прогнать",
    "индекс на таблицу поставить", "кверь тормозит как ускорить",
    "форен ключ не срабатывает", "дамп базы слепить",
    "мои поставки гет запрос", "заказ пост создать", "статус патчем поменять",
    "удалить из корзины товар", "bearer токен в авторизации",
    "роли админ/менеджер/клиент разница", "менеджеру роль поднять",
    "забыл пароль восстановить", "двухфакторку включить", "логин не пускает",
    "продажи за месяц отчёт", "поставки в эксель выгрузить", "остатки на графике",
    "лидеры поставщиков по весу", "средний чек посчитать",
    "1С подружить с платформой", "телеграм уведомления настроить",
    "рассылка по клиентам", "апи ключ сгенерить", "вебхук на заказы",
    "502 гейтвей что за фигня", "докер не заводится", "память жрёт python",
    "csrf токен невалид", "запросы таймаутят"
]

print("🔥 МОЩНЫЙ B2B ИИ ТЕСТИРУЕТСЯ:")
print("=" * 80)





print("🔥 РЕАЛЬНАЯ ТОЧНОСТЬ:")
success = 0
for i, question in enumerate(tests):
    answer, nn_dist, best_idx = ai.generate_answer(question)

    is_correct = (best_idx == i)
    if is_correct:
        success += 1

    print(f"❓ '{question[:25]}...'")
    print(f"🤖 '{answer[:35]}...'")
    print(f"📊 NN_dist={nn_dist:.3f} | idx={best_idx}/{i}")
    print(f"{'✅ ТОЧНО!' if is_correct else '❌ ПРОМАХ!'}")
    print("-" * 50)

print(f"\n🎯 ГИБРИДНАЯ ТОЧНОСТЬ: {success}/{len(tests)} = {success / len(tests) * 100:.1f}%")
