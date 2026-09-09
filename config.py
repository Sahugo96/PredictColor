"""
Конфигурационный файл проекта.
"""
import os

# Пути к данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Пользователь кладёт исходный файл (csv/xlsx/json) прямо в каталог data/
DATA_PATH = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_model.keras')

# Параметры разделения данных
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Параметры модели
MODEL_HIDDEN_LAYERS = [64, 32]              # Количество нейронов в скрытых слоях
OUTPUT_SHAPE = 3                            # Количество классов: red / black / green
COLORS = ['red', 'black', 'green']          # Порядок кодирования цветов
SEQUENCE_LENGTH = 10                        # Окно истории для LSTM
ZONES = 100                                 # Число зон по числу x (int(x*100))

# Параметры обучения
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
