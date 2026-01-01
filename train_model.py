import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset, concatenate_datasets
import json
import os
from datetime import datetime
import logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class B2BAssistantTrainer:

    def __init__(self, model_name="sberbank-ai/rugpt3medium_based_on_gpt2"):
        self.model_name = model_name
        self.base_model = None
        self.device = self._setup_device()
        logger.info(f"Используемое устройство: {self.device}")

        # Загружаем токенизатор и модель
        self.tokenizer, self.model = self.load_model_and_tokenizer()

        # Настройка для экономии памяти
        self.setup_memory_efficient_training()



    def _setup_device(self):
        """Определяем доступное устройство"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Для Mac M1/M2
        else:
            return torch.device("cpu")

    def load_model_and_tokenizer(self):
        """Загрузка модели и токенизатора"""
        logger.info(f"Загружаем модель: {self.model_name}")

        # Токенизатор
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # Устанавливаем специальные токены
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>",
            "system_token": "<|system|>",
            "context_token": "<|context|>"
        }

        # Добавляем специальные токены
        tokenizer.add_special_tokens({
            "additional_special_tokens": list(special_tokens.values())
        })

        # Модель
        model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Изменяем размер эмбеддингов для новых токенов
        model.resize_token_embeddings(len(tokenizer))

        self.base_model = self.model

        logger.info(f"Модель загружена. Размер словаря: {len(tokenizer)}")
        logger.info(f"Количество параметров модели: {sum(p.numel() for p in model.parameters()):,}")


        return tokenizer, model

    def setup_memory_efficient_training(self):
        """Настройка эффективного обучения для экономии памяти"""

        # 1. Включаем gradient checkpointing (экономит память, но замедляет)
        self.model.gradient_checkpointing_enable()

        # 2. Настраиваем LoRA (Low-Rank Adaptation) - обучаем только маленькие адаптеры
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"], # Какие слои адаптировать
            bias="none",
            use_rslora=True  # √ Квадратичное масштабирование (лучше сходимость)
        )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()  # Покажет, сколько параметров обучается

        # 3. Перемещаем модель на устройство
        self.model.to(self.device)
        logger.info("✅ LoRA настроена")

    def load_dataset(self, data_dir="data/ready_dataset"):
        """Загрузка и подготовка датасета"""
        logger.info(f"Загружаем датасет из {data_dir}")

        datasets = []

        # Загружаем все train файлы
        train_files = [
            os.path.join(data_dir, "train_1.jsonl"),
            os.path.join(data_dir, "train_2.jsonl"),
            os.path.join(data_dir, "train_3.jsonl")
        ]

        for file_path in train_files:
            if os.path.exists(file_path):
                logger.info(f"Читаем {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [json.loads(line.strip()) for line in f if line.strip()]

                texts = [item['text'] for item in lines]

                # Создаем датасет
                dataset = Dataset.from_dict({"text": texts})
                datasets.append(dataset)

        # Объединяем все train датасеты
        if datasets:
            train_dataset = concatenate_datasets(datasets)
        else:
            raise ValueError("Не найдены тренировочные данные")

        # Загружаем validation датасет
        val_file = os.path.join(data_dir, "val.jsonl")
        if os.path.exists(val_file):
            with open(val_file, 'r', encoding='utf-8') as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
            val_texts = [item['text'] for item in lines]
            val_dataset = Dataset.from_dict({"text": val_texts})
        else:
            # Если нет validation, создаем из части train
            train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']

        logger.info(f"Размер train датасета: {len(train_dataset)}")
        logger.info(f"Размер validation датасета: {len(val_dataset)}")

        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        """Токенизация текста"""
        # Токенизируем текст
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )

        # Для языкового моделирования labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def train(self, output_dir="models/b2b_assistant"):
        """Основной процесс обучения"""
        logger.info("🚀 Начинаем процесс обучения...")

        # 1. Загружаем данные
        train_dataset, val_dataset = self.load_dataset()

        # 2. Токенизируем
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        tokenized_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        # 3. Создаем data collator (для динамического паддинга)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Мы не используем masked language modeling
        )

        # 4. Настраиваем аргументы обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Количество эпох
            per_device_train_batch_size=4,  # Размер батча
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Накопление градиентов (эмулирует больший batch size)
            warmup_steps=200,
            weight_decay=0.1,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.device.type == "cuda",  # Используем mixed precision на GPU
            dataloader_num_workers=2,
            report_to="none",  # Отключаем wandb (можно включить для визуализации)
            push_to_hub=False,  # Не загружаем на Hugging Face Hub
            #gradient_checkpointing=True,  # Экономия памяти
            optim="adamw_torch",  # Оптимизатор
            learning_rate=2e-4,  # Скорость обучения
            lr_scheduler_type="cosine",  # Тип шедулера
            max_grad_norm=1.0,  # Клиппинг градиентов
        )

        # 5. Создаем тренер
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # 6. Запускаем обучение
        logger.info("Начинаем обучение...")
        trainer.train()

        # 7. Сохраняем модель
        logger.info("Сохранение модели...")

        # Сохраняем полную модель
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Сохраняем модель в формате для инференса
        self.save_model_for_inference(output_dir)

        logger.info(f"✅ Обучение завершено! Модель сохранена в {output_dir}")

        return trainer


    def save_model_for_inference(self, output_dir):
        """Сохраняем модель для инференса (LoRA уже включена)"""
        inference_path = os.path.join(output_dir, "inference_ready")
        os.makedirs(inference_path, exist_ok=True)

        # ✅ LoRA модель сохраняется как есть — для инференса готово!
        self.model.save_pretrained(inference_path)
        self.tokenizer.save_pretrained(inference_path)

        logger.info(f"✅ Модель для инференса: {inference_path}")
        logger.info("💡 Загрузка: AutoModelForCausalLM.from_pretrained() + peft.from_pretrained()")

    def evaluate_model(self, test_file="data/ready_dataset/test.jsonl", max_samples=100):
        """Оценка модели с метриками качества"""
        logger.info("Оценка модели...")

        if not os.path.exists(test_file):
            logger.warning(f"{test_file} не найден")
            return

        # 1. Загрузка + безопасный парсинг
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    test_data.append(data['text'])
                except:
                    continue

        test_samples = test_data[:max_samples]
        logger.info(f"Тестируем {len(test_samples)} примеров")

        results = []
        for i, text in enumerate(test_samples):
            # ✅ БЕЗОПАСНЫЙ парсинг
            question = expected = ""

            if "Пользователь:" in text:
                question_start = text.find("Пользователь:") + len("Пользователь:")
                question_end = text.find("\n", question_start)
                if question_end == -1:
                    question_end = len(text)
                question = text[question_start:question_end].strip()

            if "Ассистент:" in text:
                expected_start = text.find("Ассистент:") + len("Ассистент:")
                expected = text[expected_start:].split("\n")[0].strip()

            # Генерация
            generated = self.generate_response(question, max_length=150)

            results.append({
                "question": question[:100],
                "expected": expected,
                "generated": generated[:100]
            })

            if i % 20 == 0:
                logger.info(f"Обработано {i + 1}/{len(test_samples)}")

        # 2. Метрики качества
        from difflib import SequenceMatcher

        similarities = []
        for r in results:
            if r['expected'] and r['generated']:
                sim = SequenceMatcher(None, r['expected'].lower(), r['generated'].lower()).ratio()
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # 3. Сохранение + статистика
        eval_dir = "models/evaluation"
        os.makedirs(eval_dir, exist_ok=True)

        with open(os.path.join(eval_dir, "test_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 📊 СТАТИСТИКА
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("=" * 80)
        print(f"Всего примеров: {len(results)}")
        print(f"Среднее сходство с эталоном: {avg_similarity:.1%}")
        print("\nПримеры:")
        print("-" * 80)

        for i, r in enumerate(results[:5]):
            print(f"\n{i + 1}. Вопрос: {r['question']}")
            print(f"   Ожидалось: {r['expected']}")
            print(f"   Получено:  {r['generated']}")
            print()

        logger.info(f"✅ Оценка завершена. Среднее сходство: {avg_similarity:.1%}")

    def generate_response(self, prompt, max_length=150, temperature=0.8):
        """Генерация ответа с LoRA моделью"""
        # ✅ ВАЖНО: используем tokenizer.model_max_length если есть
        self.model.eval()

        # 1. Форматируем промпт
        formatted_prompt = f"<|user|>{prompt}<|assistant|>"
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Ограничиваем длину
        ).to(self.device)

        # 2. ✅ ПРАВИЛЬНАЯ генерация для LoRA
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=80,  # ← Новые токены, не общая длина!
                temperature=temperature,
                do_sample=True,
                top_p=0.95,  # Nucleus sampling
                top_k=40,
                repetition_penalty=1.1,  # Мягче
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Декодируем ПОЛНЫЙ ответ
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4. ✅ БЕЗОПАСНО извлекаем ответ ассистента
        if "<|assistant|>" in full_text:
            assistant_response = full_text.split("<|assistant|>")[-1].strip()
        else:
            assistant_response = full_text[len(formatted_prompt):].strip()

        # 5. Убираем префиксы типа "Ассистент:"
        response = assistant_response.split("\n")[0].strip()
        if response.startswith("Ассистент:"):
            response = response.replace("Ассистент:", "").strip()

        return response[:250]  # Ограничиваем длину






# Функция для интерактивного тестирования
def interactive_test(model_path="models/b2b_assistant"):
    """Интерактивное тестирование модели"""

    print("🤖 ЗАГРУЗКА LoRA МОДЕЛИ...")

    # ✅ 1. БАЗОВАЯ модель
    base_model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"  # ruGPT-3Small
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)

    # ✅ 2. НАШИ LoRA веса
    model = PeftModel.from_pretrained(base_model, model_path)

    # ✅ 3. Токены (если добавляли)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"✅ LoRA модель загружена на {device}")
    print("\n" + "=" * 70)
    print("🤖 B2B AI АССИСТЕНТ (ruGPT-3Small + LoRA)")
    print("=" * 70)
    print("💡 Напишите 'выход' для завершения")

    while True:
        try:
            question = input("\n🧑 Вы: ").strip()
            if question.lower() in ['выход', 'exit', 'quit']:
                print("👋 До свидания!")
                break
            if not question:
                continue

            # ✅ Правильный промпт
            prompt = f"<|user|>{question}<|assistant|>"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # ✅ Оптимизированная генерация
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=100,  # Только новые токены!
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,  # Мягче
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # ✅ Извлечение ответа
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            if "<|assistant|>" in full_text:
                answer = full_text.split("<|assistant|>")[-1].strip()
            else:
                answer = full_text[len(prompt):].strip()

            print(f"🤖 Ассистент: {answer.split('\n')[0]}")  # Первая строка

        except KeyboardInterrupt:
            print("\n👋 Завершение...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    # ✅ Освобождаем память
    del model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None





# Основная функция
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение B2B AI ассистента")
    parser.add_argument("--mode", choices=["train", "test", "interactive"],
                        default="train", help="Режим работы")
    parser.add_argument("--model_path", default="models/b2b_assistant",
                        help="Путь к модели")
    parser.add_argument("--data_dir", default="data/ready_dataset",
                        help="Путь к данным")

    args = parser.parse_args()

    if args.mode == "train":
        # Обучение модели
        trainer = B2BAssistantTrainer()
        trainer.train(output_dir=args.model_path)

        # Оценка после обучения
        trainer.evaluate_model()

        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Модель сохранена в: {args.model_path}")
        print("📊 Для тестирования запустите: python train_colab.py --mode interactive")
        print("=" * 80)


    elif args.mode == "test":

        trainer = B2BAssistantTrainer(args.model_path)  # ✅ Загрузит LoRA

        trainer.evaluate_model(os.path.join(args.data_dir, "test.jsonl"))

    elif args.mode == "interactive":
        # Интерактивный режим
        interactive_test(args.model_path)
