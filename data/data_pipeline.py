import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import os

class DataPipeline:
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
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

        print(f"✅ Загружено {len(pairs)} B2B пар")
        return pairs

    def load_sberquad(self):
        """Загружаем SberQuAD с Hugging Face"""
        print("📂 Загружаем SberQuAD...")

        dataset = load_dataset("kuznetsoffandrey/sberquad")

        texts = []
        # SberQuAD имеет формат: {'id', 'title', 'context', 'question', 'answers'}
        for split in ['train', 'validation']:
            if split in dataset:
                for item in dataset[split]:
                    if item.get('question') and (item.get('answers') or item.get('answer')):
                        answer = item['answers']['text'][0] if item['answers']['text'] else item['answer']['text'][0] if item['answer']['text'] else 'Информация содержится в тексте выше.'
                        text = f"Пользователь: {item['question']}\nАссистент: {answer}"
                        texts.append(text)

        print(f"✅ Загружено {len(texts)} примеров из SberQuAD")
        return texts

    def load_russian_superglue(self, datasets_path):
        """Загружаем все датасеты Russian SuperGLUE"""
        print("📂 Загружаем Russian SuperGLUE...")

        texts = []
        glue_path = Path(datasets_path)

        target_folders = ['DaNetQA', 'TERRa', 'PARus', 'MuSeRC', 'RuCoS']

        for folder in target_folders:
            folder_path = glue_path / folder
            if folder_path.exists():
                print(f"  Обрабатываем {folder}...")

                for train_file in folder_path.rglob('train.jsonl'):
                    with open(train_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
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
                text = passage.get('text', '')
                query = qas[0].get('query', '')

                highlights_raw = passage.get('text', '').partition('@highlight')[2]  # всё после @highlight
                highlights_text = '\n'.join([h.strip() for h in highlights_raw.split('@highlight') if h.strip()])

                # Правильный ответ из entities
                entities = data.get('entities', [])
                correct_answer = "Неизвестно"
                if entities:
                    start, end = entities[0]['start'], entities[0]['end']

                    correct_answer = text[start:end].strip()

                return f"Пользователь: {query}\nКонтекст: {text}\nHighlights:\n{highlights_text}\nАссистент: {correct_answer}"
            else:
                return None











