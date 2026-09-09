"""
Основной файл приложения для предсказания событий с использованием нейронной сети.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from modules.event_predictor import EventPredictorModel
from utils.data_loader import DataLoader
from utils.preprocessing import Preprocessor
import config

NAME = {i: c for i, c in enumerate(config.COLORS)}


def accuracy(y_idx, pred_idx):
    """Доля совпадений двух массивов меток."""
    return float(np.mean(y_idx == pred_idx))


def main():
    """
    Основная функция приложения.

    В данных обнаружена детерминированная связь: цвет выпавшего исхода
    практически полностью задаётся числом x из колонки «Число» —
    зоной int(x*100). Поэтому:

      1) зонная таблица (правило по x)      -> ~97% точности
      2) нейросеть FFN по one-hot зоны     -> ~97% точности
      3) LSTM по истории цветов            -> ~53% (потолок: переходы
                                              почти независимы)
    """
    print("Инициализация проекта для предсказания событий...")

    # Загрузка данных (сырой файл: xlsx/csv/json из data/)
    data_loader = DataLoader(config.DATA_PATH)
    raw_data = data_loader.load_data(raw=True)

    # Предобработка данных: чистка, кодирование цветов, зоны по числу x
    preprocessor = Preprocessor(
        sequence_length=config.SEQUENCE_LENGTH,
        colors=config.COLORS,
        n_zones=config.ZONES
    )
    df = preprocessor.process(raw_data)
    print(f"Записей спинов: {len(df)}")
    print("Распределение цветов:\n" + df['цвет'].value_counts().to_string())

    # Метки цветов (0/1/2) и зоны числа x
    y_idx = preprocessor.encode_targets(df['цвет'].values)
    zones = preprocessor.zone_index(df['число'].values)

    split = int(len(df) * (1 - config.TEST_SIZE))
    z_tr, z_te = zones[:split], zones[split:]
    y_tr, y_te = y_idx[:split], y_idx[split:]
    print(f"Train: {split}, Test: {len(df) - split}")

    # ---------------------------------------------------------------
    # 1) Зонная таблица: для каждой зоны берём цвет большинства на train
    # ---------------------------------------------------------------
    table = preprocessor.zone_table(z_tr, y_tr)
    pred_tab = preprocessor.predict_by_table(table, z_te)
    acc_tab = accuracy(y_te, pred_tab)
    print(f"\nЗонная таблица (int(x*100)):     точность = {acc_tab:.4f}")

    # ---------------------------------------------------------------
    # 2) Нейросеть FFN: вход one-hot зоны -> цвет
    # ---------------------------------------------------------------
    Xz_tr = preprocessor.zone_onehot(z_tr)
    Xz_te = preprocessor.zone_onehot(z_te)
    Y_tr = preprocessor.to_onehot(y_tr)
    Y_te = preprocessor.to_onehot(y_te)

    model = EventPredictorModel(
        input_shape=(config.ZONES,),
        hidden_layers=config.MODEL_HIDDEN_LAYERS,
        output_shape=config.OUTPUT_SHAPE,
        learning_rate=config.LEARNING_RATE,
        kind='ffn'
    )
    history = model.train(Xz_tr, Y_tr, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    loss_nn, acc_nn = model.evaluate(Xz_te, Y_te)
    print(f"Нейросеть FFN по зоне x:         точность = {acc_nn:.4f} (loss={loss_nn:.4f})")

    # ---------------------------------------------------------------
    # 3) LSTM по истории цветов (для сравнения)
    # ---------------------------------------------------------------
    X_seq, Y_seq = preprocessor.prepare_xy(df)
    X_ltr, X_lte, Y_ltr, Y_lte = preprocessor.train_test_split(
        X_seq, Y_seq, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    model_l = EventPredictorModel(
        input_shape=X_ltr.shape[1:],
        hidden_layers=config.MODEL_HIDDEN_LAYERS,
        output_shape=config.OUTPUT_SHAPE,
        learning_rate=config.LEARNING_RATE,
        kind='lstm'
    )
    model_l.train(X_ltr, Y_ltr, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    loss_l, acc_l = model_l.evaluate(X_lte, Y_lte)
    print(f"LSTM по истории цветов:          точность = {acc_l:.4f} (loss={loss_l:.4f})")

    # Базовый уровень: всегда предсказываем самый частый цвет из трейна
    gbl = int(np.bincount(y_tr, minlength=3).argmax())
    base = accuracy(y_te, np.full(len(y_te), gbl))
    print(f"Baseline (всегда «{NAME[gbl]}»):       точность = {base:.4f}")

    # Примеры предсказаний нейросети по зонам
    pred_idx = np.argmax(model.predict(Xz_te[:10]), axis=1)
    print("\nПримеры предсказаний FFN на тесте:")
    for i, (p, a) in enumerate(zip(pred_idx, y_te[:10])):
        mark = "+" if p == a else "-"
        print(f"  {mark} прогноз={NAME[p]:<6} факт={NAME[a]}")

    # Сохранение модели и обработанных данных
    model.save(config.MODEL_SAVE_PATH)
    data_loader.save_processed_data(df, 'processed_colors.csv')

    print(f"\nМодель сохранена в {config.MODEL_SAVE_PATH}")
    print("Проект отработал успешно!")


if __name__ == "__main__":
    main()
