import argparse
import logging
import pickle
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import torch
from torch.utils.data import random_split
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
# Эта библиотека не используется напрямую, но нужна для интеграции с Hub
from huggingface_hub import HfApi # Импорт для ясности, но Trainer сам использует хаб

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Загружена конфигурация из {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Файл конфигурации не найден: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке конфигурации: {e}")
        raise


# def load_and_prepare_data(config: Dict[str, Any]) -> (Dataset, Dataset):
#     """
#     Loads training data from CSV, prepares training examples,
#     and splits into train/validation sets.
#     """
#     data_config = config['data']
#     train_path = Path(data_config['train_path'])
#     # eval_path = Path(data_config['eval_pickle_path']) # Закомментировано, т.к. eval берется из split

#     try:
#         logging.info(f"Загрузка данных для обучения из: {train_path}")
#         df_train = pd.read_csv(train_path)
#         # Basic cleaning: remove rows where query or passage might be NaN/empty
#         df_train.dropna(
#             subset=[data_config['train_query_column'], data_config['train_passage_column']],
#             inplace=True
#         )
#         df_train = df_train.astype(str) # Ensure string type
#         logging.info(f"Загружено {len(df_train)} строк для обучения.")
#     except FileNotFoundError:
#         logging.error(f"Файл данных для обучения не найден: {train_path}")
#         raise
#     except Exception as e:
#         logging.error(f"Ошибка при загрузке данных для обучения: {e}")
#         raise

#     # --- Prepare Training Examples ---
#     # Префиксы больше не добавляются здесь, т.к. MNRLoss обычно не требует их,
#     # но если они нужны вашей модели, добавьте их обратно.
#     # query_prefix = data_config.get('query_prefix', '')
#     # passage_prefix = data_config.get('passage_prefix', '')

#     train_examples_list = []
#     for _, row in df_train.iterrows():
#         query = row[data_config['train_query_column']]
#         passage = row[data_config['train_passage_column']]
#         train_examples_list.append({'anchor': query, 'positive': passage})

#     train_dataset_full = Dataset.from_list(train_examples_list)
#     logging.info(f"Создан полный датасет для обучения: {len(train_dataset_full)} примеров.")

#     # --- Split Data ---
#     validation_size = config['training'].get('validation_split_size', 0.05)
#     if validation_size > 0:
#         logging.info(f"Разделение данных на обучающую и валидационную выборки (доля валидации: {validation_size})")
#         dataset_dict = train_dataset_full.train_test_split(test_size=validation_size, seed=config['training'].get('seed', 42))
#         train_dataset = dataset_dict["train"] # Это будет datasets.Dataset
#         eval_dataset = dataset_dict["test"]  # Это будет datasets.Dataset
#         logging.info(f"Размер обучающей выборки: {len(train_dataset)}")
#         logging.info(f"Размер валидационной выборки: {len(eval_dataset)}")
#     else:
#         logging.info("Валидационная выборка не создается (validation_split_size <= 0).")
#         train_dataset = train_dataset_full
#         eval_dataset = None # Нет валидационного датасета

#     # --- Загрузка данных для оценки (если используется InformationRetrievalEvaluator) ---
#     # Этот блок закомментирован, так как текущий код использует простой eval_dataset
#     # Если вам нужен IR Evaluator, раскомментируйте и настройте его
#     # try:
#     #     with open(eval_path, 'rb') as f:
#     #         eval_data = pickle.load(f)
#     #     # Validate evaluation data structure
#     #     if not all(k in eval_data for k in ['queries', 'corpus', 'relevant_docs']):
#     #          raise ValueError("Evaluation data pickle must contain 'queries', 'corpus', and 'relevant_docs' keys.")
#     #     logging.info(f"Загружены данные для IR оценки из {eval_path}")
#     # except FileNotFoundError:
#     #     logging.warning(f"Файл данных для IR оценки не найден: {eval_path}. IR Evaluator не будет использован.")
#     #     eval_data = None
#     # except Exception as e:
#     #     logging.error(f"Ошибка при загрузке данных для IR оценки: {e}")
#     #     eval_data = None

#     return train_dataset, eval_dataset #, eval_data # Возвращаем eval_data если используем IR Evaluator


def load_and_prepare_data(config: Dict[str, Any]) -> (Dataset, Dataset):
    """Загружает данные для q2q и q2p раздельно и объединяет с меткой типа."""
    data_config = config['data']
    
    # Загрузка query2query данных
    q2q_df = pd.read_csv(data_config['q2q_train_path'])
    q2q_df.dropna(subset=[data_config['train_query_column'], data_config['train_passage_column']], inplace=True)
    q2q_df = q2q_df.astype(str)
    
    # Загрузка query2passage данных
    q2p_df = pd.read_csv(data_config['q2p_train_path'])
    q2p_df.dropna(subset=[data_config['train_query_column'], data_config['train_passage_column']], inplace=True)
    q2p_df = q2p_df.astype(str)
    
    # Создание примеров с префиксами и меткой типа
    q2q_examples = []
    for _, row in q2q_df.iterrows():
        # query = f"query: {row[data_config['train_query_column']]}"
        # passage = f"query: {row[data_config['train_passage_column']]}"
        q2q_examples.append({'anchor': query, 'positive': passage, 'task_type': 'q2q'})
    
    q2p_examples = []
    for _, row in q2p_df.iterrows():
        # query = f"query: {row[data_config['train_query_column']]}"
        # passage = f"passage: {row[data_config['train_passage_column']]}"
        q2p_examples.append({'anchor': query, 'positive': passage, 'task_type': 'q2p'})
    
    # Объединение датасетов
    combined_dataset = Dataset.from_list(q2q_examples + q2p_examples)
    
    # Разделение на train/eval
    validation_size = config['training'].get('validation_split_size', 0.05)
    if validation_size > 0:
        dataset_dict = combined_dataset.train_test_split(test_size=validation_size, seed=config['training'].get('seed', 42))
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    else:
        train_dataset = combined_dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset


from torch.utils.data import BatchSampler
import numpy as np

class TaskTypeBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Группируем индексы по task_type
        self.q2q_indices = [i for i, ex in enumerate(dataset) if ex['task_type'] == 'q2q']
        self.q2p_indices = [i for i, ex in enumerate(dataset) if ex['task_type'] == 'q2p']
        
        self._generate_batches()
    
    def _generate_batches(self):
        # Перемешиваем внутри каждой группы
        if self.shuffle:
            np.random.shuffle(self.q2q_indices)
            np.random.shuffle(self.q2p_indices)
        
        # Создаем батчи для каждой группы
        self.batches = []
        for group in [self.q2q_indices, self.q2p_indices]:
            for i in range(0, len(group), self.batch_size):
                self.batches.append(group[i:i+self.batch_size])
        
        # Перемешиваем порядок батчей
        if self.shuffle:
            np.random.shuffle(self.batches)
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


def initialize_model_and_loss(config: Dict[str, Any]) -> (SentenceTransformer, torch.nn.Module):
    """Initializes the Sentence Transformer model and the loss function."""
    model_config = config['model']
    model_name = model_config['base_model_name_or_path']

    try:
        logging.info(f"Инициализация модели: {model_name}")
        model = SentenceTransformer(model_name)
        logging.info("Модель успешно инициализирована.")
    except Exception as e:
        logging.error(f"Ошибка при инициализации модели {model_name}: {e}")
        raise

    # MultipleNegativesRankingLoss подходит для пар запрос-пассаж
    # Ожидает Dataset с колонками 'anchor' и 'positive'
    # Негативные примеры автоматически сэмплируются из батча
    loss_name = config['training'].get('loss_function', 'MultipleNegativesRankingLoss')
    if loss_name == 'MultipleNegativesRankingLoss':
        logging.info("Используется функция потерь: MultipleNegativesRankingLoss")
        loss = losses.MultipleNegativesRankingLoss(model)
    # Добавьте другие функции потерь при необходимости
    # elif loss_name == 'SomeOtherLoss':
    #    loss = losses.SomeOtherLoss(model, ...)
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss_name}")

    return model, loss


def setup_training(
    config: Dict[str, Any],
    model: SentenceTransformer,
    loss: torch.nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset # Может быть None
    # eval_data: Dict[str, Any] # Раскомментируйте, если используете IR Evaluator
) -> SentenceTransformerTrainer: # Убрали IR Evaluator из возвращаемого значения
    """Sets up the training arguments and the trainer."""
    training_config = config['training']
    # eval_config = config['evaluation'] # Закомментировано, так как IR Evaluator не используется

    # --- Setup Evaluator (Закомментировано) ---
    # ir_evaluator = None
    # if eval_data:
    #     logging.info("Настройка InformationRetrievalEvaluator.")
    #     ir_evaluator = InformationRetrievalEvaluator(
    #         queries=eval_data['queries'],
    #         corpus=eval_data['corpus'],
    #         relevant_docs=eval_data['relevant_docs'],
    #         name=eval_config.get('eval_name', "ir_eval"),
    #         # accuracy_at_k=eval_config.get('ir_accuracy_at_k', [1, 3, 5]), # Использует mrr, map, ndcg по умолчанию
    #         precision_recall_at_k=eval_config.get('ir_precision_recall_at_k', [1, 5, 10]),
    #         map_at_k=eval_config.get('ir_map_at_k', [10, 100]),
    #         ndcg_at_k=eval_config.get('ir_ndcg_at_k', [10, 100]),
    #         show_progress_bar=True,
    #     )
    # else:
    #     logging.info("Данные для IR Evaluator не предоставлены, он не будет использоваться.")

    # --- Setup Training Arguments ---
    try:
        logging.info("Настройка аргументов обучения.")
        output_dir = Path(training_config['output_dir'])
        logging_dir = Path(training_config.get('logging_dir', output_dir / 'logs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        # Map batch sampler string from config to enum
        # sampler_str = training_config.get("batch_sampler", "NO_DUPLICATES").upper()
        # # Убедимся, что значение существует в BatchSamplers
        # try:
        #     batch_sampler = BatchSamplers[sampler_str]
        #     logging.info(f"Используется BatchSampler: {batch_sampler}")
        # except KeyError:
        #     logging.warning(f"Неверный batch_sampler '{sampler_str}'. Используется NO_DUPLICATES.")
        #     batch_sampler = BatchSamplers.NO_DUPLICATES

        # Определение стратегии оценки и сохранения
        eval_strategy = training_config.get('eval_strategy', 'epoch' if eval_dataset else 'no')
        save_strategy = training_config.get('save_strategy', 'epoch' if eval_dataset else 'steps') # Сохраняем по шагам, если нет eval
        save_steps = training_config.get('save_steps', 500) # Шаг сохранения, если save_strategy='steps'

        load_best = training_config.get('load_best_model_at_end', bool(eval_dataset)) # Загружать лучшую, если есть оценка
        metric_for_best = training_config.get('metric_for_best_model', None)
        if load_best and not metric_for_best and eval_dataset:
             # Пытаемся установить метрику по умолчанию для MNRLoss, если она не задана
             # Обычно trainer сам выбирает что-то вроде eval_loss
             logging.warning("load_best_model_at_end=True, но metric_for_best_model не задана. Trainer попытается использовать eval_loss.")
             # metric_for_best = 'eval_loss' # Можно задать явно, если нужно

        # --- Hugging Face Hub Parameters ---
        push_to_hub = training_config.get('push_to_hub', False)
        hub_model_id = training_config.get('hub_model_id', None)
        hub_strategy = training_config.get('hub_strategy', 'end') # 'end', 'every_save', 'checkpoint'
        hub_private = training_config.get('hub_private_repo', False)

        if push_to_hub and not hub_model_id:
            hub_model_id = output_dir.name # Использовать имя папки вывода как ID модели, если не указано
            logging.warning(f"hub_model_id не указан, используется имя папки вывода: {hub_model_id}")

        batch_size = training_config.get('per_device_train_batch_size', 64)
        train_sampler = TaskTypeBatchSampler(
            train_dataset, 
            batch_size=batch_size,
            shuffle=config['training'].get('shuffle', True)
        )

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(Path(training_config['output_dir'])),
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
            learning_rate=training_config.get('learning_rate', 2e-5),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.0),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', False),
            # batch_sampler=batch_sampler,
            data_loader_kwargs={
                'sampler': train_sampler,
                'batch_size': 1,
                'drop_last': False
            }
            eval_strategy=training_config.get('eval_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 25),
            save_strategy=training_config.get('save_strategy', 'steps'),
            save_steps=training_config.get('save_steps', 475),
            save_total_limit=training_config.get('save_total_limit', 1),
            load_best_model_at_end=training_config.get('load_best_model_at_end', False),
            metric_for_best_model=training_config.get('metric_for_best_model', None),
            logging_steps=training_config.get('logging_steps', 10),
            logging_dir=str(Path(training_config.get('logging_dir', str(Path(training_config['output_dir']) / 'logs')))),
            report_to=training_config.get('report_to', "tensorboard").split(','),

            # --- Интеграция с Hugging Face Hub ---
            push_to_hub=training_config.get('push_to_hub'),
            hub_model_id=training_config.get('hub_model_id'),
            hub_strategy=training_config.get('hub_strategy'),
            hub_token=os.environ.get('HF_TOKEN') # Можно передать токен явно, но обычно лучше через login/env var
        )
        logging.info("Аргументы обучения успешно созданы.")
        if push_to_hub:
            logging.info(f"Модель будет загружена в Hugging Face Hub: repo_id='{hub_model_id}', strategy='{hub_strategy}', private={hub_private}")
            logging.info("Убедитесь, что вы вошли в систему через `huggingface-cli login` или установили переменную окружения HF_TOKEN.")

    except Exception as e:
        logging.error(f"Ошибка при настройке аргументов обучения: {e}")
        raise

    # --- Setup Trainer ---
    logging.info("Инициализация SentenceTransformerTrainer.")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Передаем eval_dataset (может быть None)
        loss=loss,
        # evaluator=ir_evaluator, # Передаем IR Evaluator, если он есть
        # compute_metrics=compute_metrics_callback, # Можно добавить свою функцию метрик для eval_dataset
    )
    logging.info("SentenceTransformerTrainer успешно инициализирован.")

    return trainer #, ir_evaluator # Возвращаем evaluator, если он используется


def main(config_path: str):
    """Main function to orchestrate the training process."""
    logging.info("--- Начало процесса обучения ---")

    try:
        # 1. Загрузка конфигурации
        config = load_config(config_path)

        # 2. Загрузка и подготовка данных
        # train_dataset, eval_dataset, eval_data = load_and_prepare_data(config) # Если с IR Eval
        train_dataset, eval_dataset = load_and_prepare_data(config) # Если без IR Eval

        # 3. Инициализация модели и функции потерь
        model, loss = initialize_model_and_loss(config)

        # 4. Настройка обучения (Аргументы, Тренер)
        # trainer, ir_evaluator = setup_training(config, model, loss, train_dataset, eval_dataset, eval_data) # Если с IR Eval
        trainer = setup_training(config, model, loss, train_dataset, eval_dataset) # Если без IR Eval

        # 5. Обучение модели
        logging.info("--- Начало обучения модели ---")
        try:
            train_result = trainer.train()
            logging.info("--- Обучение модели завершено ---")
            logging.info(f"Результаты обучения: {train_result.metrics}")

            # Сохраняем метрики обучения
            metrics_save_path = Path(config['training']['output_dir']) / "train_results.json"
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            logging.info(f"Метрики обучения сохранены в: {metrics_save_path}")

        except Exception as e:
            logging.error(f"Ошибка во время обучения: {e}")
            raise

        # 6. Сохранение финальной (или лучшей) модели локально
        # Если load_best_model_at_end=True, trainer уже загрузил лучшую модель.
        # trainer.save_model() сохранит текущую модель тренера (которая может быть лучшей)
        # в output_dir. push_to_hub также вызывается здесь при стратегии 'end'.
        # Дополнительное сохранение может быть полезно, если нужно сохранить
        # в другое место, отличное от output_dir/hub_model_id.
        final_save_path_str = config['model'].get('save_path', None)
        if final_save_path_str:
            final_save_path = Path(final_save_path_str)
            final_save_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Сохранение финальной модели локально в: {final_save_path}")
            # model.save(str(final_save_path)) # Можно так, если trainer обновил объект model
            trainer.save_model(str(final_save_path)) # Или так, чтобы точно сохранить модель из тренера
            logging.info("Финальная модель успешно сохранена локально.")
        else:
            logging.info("Локальный путь для сохранения финальной модели (model.save_path) не указан, сохранение пропускается.")
            logging.info(f"Модель (и чекпоинты) доступны в: {config['training']['output_dir']}")

        # Важно: Если push_to_hub=True и hub_strategy='end', загрузка на Hub
        # происходит автоматически в конце trainer.train() или при вызове trainer.push_to_hub()
        # Явный вызов push_to_hub здесь не обязателен, если стратегия 'end'.
        # if config['training'].get('push_to_hub', False):
        #     try:
        #         logging.info(f"Попытка загрузки модели на Hugging Face Hub: {config['training']['hub_model_id']}")
        #         # trainer.push_to_hub() # Можно вызвать явно, если нужно
        #         logging.info("Модель успешно загружена на Hub (или была загружена автоматически).")
        #     except Exception as e:
        #         logging.error(f"Ошибка при загрузке модели на Hugging Face Hub: {e}")


    except FileNotFoundError as e:
        logging.error(f"Ошибка: Необходимый файл не найден. {e}")
    except ValueError as e:
        logging.error(f"Ошибка в данных или конфигурации: {e}")
    except Exception as e:
        logging.error(f"Непредвиденная ошибка в главном процессе: {e}", exc_info=True) # Добавляем traceback
    finally:
        logging.info("--- Процесс обучения завершен ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sentence Transformer Embedding Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., config.yaml)",
    )
    args = parser.parse_args()

    main(args.config)