"""
Модель нейронной сети для предсказания событий.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # тихий TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

class EventPredictorModel:
    """
    Нейросетевая модель для классификации исхода рулетки.

    kind='ffn'  — фидфорвард-сеть по признаку "зона x" (int(x*100)):
                  на входе one-hot из N_ZONES зон, на выходе softmax по цветам.
                  Даёт ~97% точности на данных, т.к. цвет детерминирован зоной.
    kind='lstm' — рекуррентная сеть по истории цветов (окно из sequence_length
                  спинов). Статистический потолок такого прогноза ~53%
                  (переходы практически независимы).
    """

    def __init__(self, input_shape, hidden_layers, output_shape,
                 learning_rate=0.001, kind='ffn'):
        """
        Инициализация модели.

        Args:
            input_shape: Форма входных данных
                - ffn:  (n_zones,)
                - lstm: (sequence_length, features)
            hidden_layers: Список с количеством нейронов в скрытых слоях
            output_shape: Число классов на выходе
            learning_rate: Скорость обучения
            kind: 'ffn' или 'lstm'
        """
        self.learning_rate = learning_rate
        self.kind = kind
        self.model = self._build_model(input_shape, hidden_layers, output_shape)

    def _build_model(self, input_shape, hidden_layers, output_shape):
        """
        Построение архитектуры модели.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))

        if self.kind == 'lstm':
            # Рекуррентная ветка: LSTM по истории спинов
            model.add(LSTM(hidden_layers[0], return_sequences=len(hidden_layers) > 1))
            model.add(Dropout(0.2))
            for i in range(1, len(hidden_layers)):
                return_sequences = i < len(hidden_layers) - 1
                model.add(LSTM(hidden_layers[i], return_sequences=return_sequences))
                model.add(Dropout(0.2))
        else:
            # Плотная ветка: классификация по зоне числа x
            for units in hidden_layers:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(0.2))

        # Выходной слой (softmax по классам цветов)
        model.add(Dense(output_shape, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=0):
        """
        Обучение модели.

        Args:
            X: Входные данные
            y: Целевые значения
            epochs: Количество эпох обучения
            batch_size: Размер батча
            validation_split: Доля данных для валидации
            verbose: Уровень логирования (0/1/2)

        Returns:
            История обучения
        """
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history

    def predict(self, X):
        """
        Предсказание следующего события.

        Args:
            X: Входные данные

        Returns:
            Предсказанные вероятности для каждого класса
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Оценка модели.

        Args:
            X: Входные данные
            y: Целевые значения

        Returns:
            Кортеж (loss, accuracy)
        """
        return self.model.evaluate(X, y)

    def save(self, path):
        """
        Сохранение модели.

        Args:
            path: Путь для сохранения модели (.keras или .h5)
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """
        Загрузка модели.

        Args:
            path: Путь к сохраненной модели

        Returns:
            Загруженная модель
        """
        model = tf.keras.models.load_model(path)
        return model
