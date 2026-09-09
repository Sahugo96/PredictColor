"""
Модуль для предобработки данных рулетки.
Задача: по истории выпавших цветов (red / black / green) предсказать следующий цвет.
"""
import numpy as np
import pandas as pd


class Preprocessor:
    """
    Предобработка сырого xlsx-файла рулетки и подготовка окон для LSTM.
    """

    def __init__(self, sequence_length=10, colors=('red', 'black', 'green'), n_zones=100):
        self.sequence_length = sequence_length
        self.colors = list(colors)
        self.color_map = {c: i for i, c in enumerate(self.colors)}
        self.n_zones = n_zones

    def process(self, raw):
        """
        Очистка сырого DataFrame: берутся только значимые колонки (A-F)
        и строки-записи спинов (фильтр по цвету).

        Args:
            raw: DataFrame, прочитанный с header=None (включая 2 строки-шапки).

        Returns:
            DataFrame с колонками: дата, цвет, число, сессия, спин, флаг
        """
        df = raw.iloc[2:, :6].copy()  # срезаем шапки (2 строки), оставляем A-F
        df.columns = ['дата', 'цвет', 'число', 'сессия', 'спин', 'флаг']
        df = df[df['цвет'].isin(self.colors)].copy()
        df = df.dropna(subset=['дата'])
        # «Число» приводим к float; некорректные значения -> NaN
        df['число'] = pd.to_numeric(df['число'], errors='coerce')
        df.reset_index(drop=True, inplace=True)
        return df

    def encode_targets(self, colors):
        """Цвета -> целочисленные метки классов."""
        return np.array([self.color_map[c] for c in colors], dtype=np.int64)

    def to_onehot(self, encoded):
        """
        Метки -> one-hot представление.
        Для вектора (n,) -> (n, C); для матрицы окон (n, seq) -> (n, seq, C).
        """
        n_classes = len(self.colors)
        if encoded.ndim == 1:
            out = np.zeros((len(encoded), n_classes), dtype=np.float32)
            out[np.arange(len(encoded)), encoded] = 1.0
            return out
        out = np.zeros(encoded.shape + (n_classes,), dtype=np.float32)
        rows = np.arange(encoded.shape[0])[:, None]
        cols = np.arange(encoded.shape[1])[None, :]
        out[rows, cols, encoded] = 1.0
        return out

    def create_sequences(self, encoded):
        """
        Создание последовательностей для временного ряда:
        X[i] = окно из sequence_length цветов, y[i] = следующий цвет.
        """
        sequences, targets = [], []
        for i in range(len(encoded) - self.sequence_length):
            sequences.append(encoded[i:i + self.sequence_length])
            targets.append(encoded[i + self.sequence_length])
        return np.array(sequences), np.array(targets)

    def prepare_xy(self, df):
        """
        Полный конвейер: очищенный DataFrame -> (X, y) для обучения.
        X: (n, sequence_length, num_colors), y: (n, num_colors).
        """
        encoded = self.encode_targets(df['цвет'].values)
        X, y = self.create_sequences(encoded)
        return self.to_onehot(X), self.to_onehot(y)

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Разделение на обучающую и тестовую выборки.
        Без перемешивания (shuffle=False): данные — временной ряд,
        тестируемся только на «будущем» (последние test_size окон).
        """
        split = int(len(X) * (1 - test_size))
        return X[:split], X[split:], y[:split], y[split:]

    # ------------------------------------------------------------------
    #  Зонные признаки: цвет связан с числом x через таблицу зон int(x*100)
    # ------------------------------------------------------------------

    def zone_index(self, xs):
        """
        xs: массив чисел x (0..1) -> зона = int(x*100), NaN -> -1.
        """
        xs = np.asarray(xs, dtype=float)
        zones = np.full(len(xs), -1, dtype=int)
        ok = ~np.isnan(xs)
        zones[ok] = np.clip(
            np.floor(xs[ok] * self.n_zones), 0, self.n_zones - 1
        ).astype(int)
        return zones

    def zone_onehot(self, zones):
        """
        Зоны (n,) -> one-hot матрица (n, n_zones); зона -1 -> нулевой вектор.
        """
        n = len(zones)
        out = np.zeros((n, self.n_zones), dtype=np.float32)
        valid = (zones >= 0) & (zones < self.n_zones)
        out[np.arange(n)[valid], zones[valid]] = 1.0
        return out

    def zone_table(self, zones_train, y_train_idx):
        """
        Строит таблицу "зона -> цвет большинства" по обучающим данным.

        Args:
            zones_train: (n,) массив зон
            y_train_idx: (n,) массив целочисленных меток цветов

        Returns:
            (np.ndarray) длины n_zones с метками цветов (0/1/2); для зон
            без обучающих примеров — глобальное большинство.
        """
        table = np.full(self.n_zones, -1, dtype=int)
        valid = (zones_train >= 0) & (zones_train < self.n_zones)
        gbl = int(np.bincount(y_train_idx, minlength=3).argmax())
        for z in range(self.n_zones):
            m = valid & (zones_train == z)
            if m.sum() == 0:
                table[z] = gbl
            else:
                table[z] = int(np.bincount(y_train_idx[m], minlength=3).argmax())
        return table

    def predict_by_table(self, table, zones):
        """
        Применяет таблицу зон: zones -> предсказанные метки цветов.
        """
        zones = np.asarray(zones, dtype=int)
        pred = np.empty(len(zones), dtype=int)
        valid = (zones >= 0) & (zones < self.n_zones)
        pred[valid] = table[zones[valid]]
        pred[~valid] = int(np.bincount(table, minlength=3).argmax())
        return pred
