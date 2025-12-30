# inspect_data.py
import json
import random
from pathlib import Path


def inspect_dataset():
    """Просмотр примеров из датасета"""

    print("🔍 Просмотр датасета...")
    print("=" * 80)
    script_dir = Path(__file__).parent

    # Читаем несколько примеров из каждого файла
    for filename in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        try:

            path = script_dir / 'ready_dataset' / filename

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]  # Первые 5 примеров

            print(f"\n📄 {filename} (первые 5 примеров):")
            print("-" * 40)

            for i, line in enumerate(lines):
                data = json.loads(line.strip())
                text = data['text']

                # Красиво выводим
                print(f"\nПример {i + 1}:")
                print(text[:200] + "..." if len(text) > 200 else text)

        except FileNotFoundError:
            print(f"Файл {filename} не найден")

    # Показываем случайные примеры
    print("\n\n🎲 Случайные примеры из train.jsonl:")
    print("-" * 40)
    train_path = script_dir / 'ready_dataset' / "train_1.jsonl"

    with open(train_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    for _ in range(3):
        random_line = random.choice(all_lines)
        data = json.loads(random_line.strip())
        text = data['text']

        # Разделяем на части для красивого вывода
        parts = text.split('\n')
        for part in parts:
            if part.startswith("Пользователь:"):
                print(f"\n🧑 {part[13:]}")
            elif part.startswith("Ассистент:"):
                print(f"🤖 {part[10:]}")
            elif part.startswith("Система:"):
                print(f"⚙️  {part[8:]}")
            elif part.startswith("Контекст:"):
                print(f"📄 {part[9:100]}...")
            else:
                print(part)

        print("-" * 40)


# Запускаем
inspect_dataset()