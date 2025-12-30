import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import os



class DataPipeline:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.all_texts = []

    def load_b2b_pairs(self, filepath):
        """Загружаем B2B пары"""
        print("📂 Загружаем B2B пары...")

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        pairs = []
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    text = f"Пользователь: {data['question']}\nАссистент: {data['answer']}"
                    pairs.append(text)
                except json.JSONDecodeError as e:
                    print(f"Ошибка в строке: '{line[:50]}...': {e}")

        print(f"✅ Загружено {len(pairs * 10)} B2B пар")
        return pairs

    def load_sberquad(self):
        """Загружаем локальный SberQuAD из jsonl (train + validation) с контекстом и логированием."""
        print("📂 Загружаем SberQuAD...")

        base_dir = Path(__file__).resolve().parent  / "sberquad"
        files = [
            base_dir / "train-00000-of-00001.jsonl",
            #base_dir / "validation-00000-of-00001.jsonl",
        ]

        texts = []
        total_lines = 0
        skipped_no_fields = 0
        skipped_empty_answer = 0

        for path in files:
            if not path.exists():
                print(f"⚠️  Файл не найден: {path}")
                continue

            print(f"  Читаем файл: {path}")
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  JSON ошибка в строке {total_lines}: {e}")
                        continue

                    context = data.get("context")
                    question = data.get("question")
                    answers = data.get("answers", {})

                    # answers может быть None или не dict
                    if not isinstance(answers, dict):
                        skipped_no_fields += 1
                        continue

                    answers_text = answers.get("text") or []

                    if not context or not question or not answers_text:
                        skipped_no_fields += 1
                        continue

                    answer = answers_text[0]
                    if not answer:
                        skipped_empty_answer += 1
                        continue

                    text = (
                        f"Пользователь: {question}\n"
                        f"Контекст: {context}\n"
                        f"Ассистент: {answer}"
                    )
                    texts.append(text)

        print(f"✅ Загружено {len(texts)} примеров из локального SberQuAD")
        print(f"ℹ️ Всего строк: {total_lines}")
        print(f"ℹ️ Пропущено из-за отсутствия полей/ответа: {skipped_no_fields}")
        print(f"ℹ️ Пропущено из-за пустого answer: {skipped_empty_answer}")
        return texts

    def load_russian_superglue(self, datasets_path):
        """Загружаем все датасеты Russian SuperGLUE (train/val/test)."""
        print("📂 Загружаем Russian SuperGLUE...")

        texts = []
        glue_path = Path(datasets_path)

        target_folders = ['DaNetQA', 'TERRa', 'PARus', 'MuSeRC', 'RuCoS', 'RUSSE']

        for folder in target_folders:
            folder_path = glue_path / folder
            if not folder_path.exists():
                continue

            print(f"  Обрабатываем {folder}...")

            if folder == 'RuCoS':
                patterns = ['val.jsonl', 'test.jsonl']
            elif folder == 'TERRa':
                patterns = ['val.jsonl', 'train.jsonl']
            elif folder == 'RUSSE':
                patterns = ['train.jsonl']
            else:
                patterns = ['train.jsonl', 'val.jsonl', 'test.jsonl']

            for pattern in patterns:
                for split_file in folder_path.rglob(pattern):
                    with split_file.open('r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            text = self.convert_superglue_to_text(data, folder)
                            if text:
                                texts.append(text)

        print(f"✅ Загружено {len(texts)} примеров из Russian SuperGLUE")
        return texts

    def convert_superglue_to_text(self, data, dataset_type):
        """Преобразуем разные форматы SuperGLUE в текстовый диалог"""
        try:
            if dataset_type == 'DaNetQA':
                question = data.get('question', '')
                passage = data.get('passage', '')
                if not passage or not question:
                    return None
                answer = "Да" if data.get('label') == True else "Нет" if data.get('label') == False else "Не знаю"
                return f"Пользователь: {question}\nКонтекст: {passage}\nАссистент: {answer}"

            elif dataset_type == 'TERRa':
                premise = data.get('premise', '')
                hypothesis = data.get('hypothesis', '')
                label = data.get('label', '')
                answer = 'Да' if label == 'entailment' else 'Нет' if label == 'not_entailment' else 'Не знаю'
                return f"Пользователь: Верно ли, что {hypothesis}?\nТекст: {premise}\nАссистент: {answer}"


            elif dataset_type == 'PARus':
                premise = data.get('premise', '')
                choice1 = data.get('choice1', '')
                choice2 = data.get('choice2', '')
                question_type = data.get('question', '')


                if question_type == "cause":
                    return f'Пользователь: Что является причиной: "{premise}"?\nВарианты: 1) {choice1}, 2) {choice2}\nАссистент: Причина: {choice1 if data.get('label') == 0 else choice2 if data.get('label') == 1 else "Не знаю"}'
                elif question_type == "effect":
                    return f'Пользователь: Что является следствием: "{premise}"?\nВарианты: 1) {choice1}, 2) {choice2}\nАссистент: Следствие: {choice1 if data.get('label') == 0 else choice2 if data.get('label') == 1 else "Не знаю"}'
                else:
                    return None


            elif dataset_type == 'MuSeRC':
                premise = data.get('premise', None)
                if premise:
                    text = premise.get('text', '')
                    questions = premise.get('questions', [])

                    formatted_questions = []
                    formatted_answers = []
                    for question in questions:
                        question_text = question.get('question', '')
                        answers = question.get('answers', [])
                        # Находим правильный ответ (label == 1)
                        correct_answer = None
                        for answer in answers:
                            if answer.get('label') == 1:
                                correct_answer = answer.get('text', '')
                                break

                        if correct_answer:
                            formatted_questions.append(question_text)
                            formatted_answers.append(correct_answer)

                    # Форматируем списки с нумерацией
                    questions_str = '\n'.join([f"{i + 1}) {q}" for i, q in enumerate(formatted_questions)])
                    answers_str = '\n'.join([f"{i + 1}) {a}" for i, a in enumerate(formatted_answers)])

                    return f'Пользователь: Текст: "{text}"\nВопросы:\n{questions_str}Ассистент: Ответы:\n{answers_str}'
                return None




            elif dataset_type == 'RuCoS':
                passage = data.get('passage', {})
                qas = data.get('qas', [])

                if not passage or not qas:
                    return None

                full_text = passage.get('text', '')
                if not full_text:
                    return None
                # Разделяем основной текст и хайлайты
                parts = full_text.split('@highlight')
                main_text = parts[0].strip()

                highlights_text = '\n'.join(p.strip() for p in parts[1:] if p.strip())
                qa = qas[0]
                query = qa.get('query', '')

                if not query or '@placeholder' not in query:
                    return None

                # Контекст вокруг placeholder

                placeholder_idx = query.index('@placeholder')
                window_left = max(0, placeholder_idx - 40)
                window_right = placeholder_idx + len('@placeholder') + 40
                around = query[window_left:window_right].lower()
                entities = data.get('entities', [])

                if not entities:
                    return None

                def safe_slice(s, start, end):
                    if 0 <= start < end <= len(s):
                        return s[start:end]
                    return ""

                candidates = []
                for ent in entities:
                    start, end = ent.get('start'), ent.get('end')

                    if start is None or end is None:
                        continue

                    ent_text = safe_slice(main_text, start, end)
                    if not ent_text:
                        ent_text = safe_slice(full_text, start, end)

                    ent_text = ent_text.strip()
                    if not ent_text:
                        continue

                    candidates.append(ent_text)

                if not candidates:
                    return None
                # Простые эвристики по окружению
                def is_person(name: str) -> bool:
                    # очень грубо: 1–3 слова с заглавной буквы
                    parts = name.split()

                    if not (1 <= len(parts) <= 3):
                        return False

                    return all(p and p[0].isupper() for p in parts)

                def pick_best_entity(candidates, around):
                    # 1) если есть "в @placeholder", "из @placeholder", "за пределами @placeholder" — география
                    if any(kw in around for kw in ['в @placeholder', 'из @placeholder', 'за пределами @placeholder']):
                        # среди кандидатов часто города/страны — 1 слово с заглавной
                        geo = [c for c in candidates if len(c.split()) <= 2 and c[0].isupper()]
                        if geo:
                            return geo[-1]  # часто последняя сущность — нужная
                    # 2) если есть "сказал @placeholder", "заявил @placeholder", "— отметил @placeholder"
                    if any(kw in around for kw in
                           ['сказал @placeholder', 'заявил @placeholder', 'отметил @placeholder']):
                        persons = [c for c in candidates if is_person(c)]
                        if persons:
                            return persons[-1]
                    # 3) общий случай — последняя осмысленная сущность
                    for c in reversed(candidates):

                        if c and not c.isdigit():
                            return c

                    return candidates[0]

                answer = pick_best_entity(candidates, around)

                if not answer:
                    return None

                return (
                    f"Пользователь: {query}\n"
                    f"Контекст: {main_text}\n"
                    f"Highlights:\n{highlights_text}\n"
                    f"Ассистент: {answer}"
                )

            elif dataset_type == 'RUSSE':

                word = data.get("word")
                s1 = data.get("sentence1")
                s2 = data.get("sentence2")
                label = data.get("label")

                if not word or not s1 or not s2 or label is None:
                        return None

                answer = "Да" if label else "Нет"

                text = (
                        f"Пользователь: В одном ли смысле употреблено слово \"{word}\" в двух предложениях?\n"
                        f"Предложение 1: {s1}\n"
                        f"Предложение 2: {s2}\n"
                        f"Ассистент: {answer}"
                )
                return text



            else:
                return None
        except Exception as e:
            print(f"Ошибка конвертации: {e}")
            return None



    def load_platform_facts(self, facts_path):
        """Загружаем факты о платформе"""
        print("📂 Загружаем факты о платформе...")

        with open(facts_path, 'r', encoding='utf-8') as f:
            facts = f.readlines()

        texts = []
        for fact in facts:
            if fact.strip():

                text = f"Система: {fact.strip()}\nАссистент: Понял, запомнил эту информацию."
                texts.append(text)

        print(f"✅ Загружено {len(texts)} фактов")
        return texts


    def create_final_dataset(self, max_lines_per_file=20000):
        """Создаем финальный датасет"""
        print("\n🎯 Создаем финальный датасет...")

        all_data = []

        b2b_pairs = self.load_b2b_pairs(f'{self.base_path}/b2b_pairs.jsonl')
        all_data.extend(b2b_pairs * 10)

        sberquad_data = self.load_sberquad()
        all_data.extend(sberquad_data[:25000])

        superglue_data = self.load_russian_superglue(f'{self.base_path}/russian_superglue')
        all_data.extend(superglue_data[:50000])

        facts_data = self.load_platform_facts(f"{self.base_path}/platform_facts.txt")
        all_data.extend(facts_data)

        import random
        random.seed(42)
        random.shuffle(all_data)

        total = len(all_data)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)

        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]

        print(f"\n📊 Статистика датасета:")
        print(f"   Всего примеров: {total:,}")
        print(f"   Train: {len(train_data):,} ({len(train_data) / total * 100:.1f}%)")
        print(f"   Validation: {len(val_data):,} ({len(val_data) / total * 100:.1f}%)")
        print(f"   Test: {len(test_data):,} ({len(test_data) / total * 100:.1f}%)")





        self.save_split(train_data, "train.jsonl")
        self.save_split(val_data, "val.jsonl")
        self.save_split(test_data, "test.jsonl")

        metadata = {
            "total_examples": total,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "sources": {
                "b2b_pairs": len(b2b_pairs),
                "sberquad": len(sberquad_data[:25000]),
                "russian_superglue": len(superglue_data[:30000]),
                "platform_facts": len(facts_data)
            }
        }

        script_dir = Path(__file__).parent  # папка с текущим скриптом
        output_path = script_dir / 'ready_dataset' / "dataset_metadata.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Финальный датасет создан!")
        print(f"   Файлы: train.jsonl, val.jsonl, test.jsonl")
        print(f"   Метаданные: dataset_metadata.json")


        return train_data, val_data, test_data

    def save_split(self, data, filename):
        """Сохраняем раздел датасета"""
        script_dir = Path(__file__).parent  # папка с текущим скриптом
        output_path_base = script_dir / 'ready_dataset'
        output_path_base.mkdir(exist_ok=True)

        # Специальная логика только для train - делим на 3 файла
        if filename == "train.jsonl":
            chunk_size = 20000
            for i, chunk in enumerate([data[j:j + chunk_size] for j in range(0, len(data), chunk_size)]):
                output_path = output_path_base / f"train_{i + 1}.jsonl"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for text in chunk:
                        # Сохраняем в простом формате: один пример = одна строка
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                print(f"📄 Сохранен train_{i + 1}.jsonl: {len(chunk)} строк")
            return

        # Для остальных файлов - как было
        output_path = output_path_base / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in data:
                # Сохраняем в простом формате: один пример = одна строка
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    pipeline = DataPipeline()
    train, val, test = pipeline.create_final_dataset()












