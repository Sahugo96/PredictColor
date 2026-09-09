"""
Модуль для загрузки данных событий.
"""
import os
import pandas as pd
import numpy as np

class DataLoader:
    """
    Класс для загрузки данных событий.
    """
    
    def __init__(self, data_path):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_path: Путь к директории с данными
        """
        self.data_path = data_path
        
    def load_data(self, filename=None, raw=False):
        """
        Загрузка данных из файла.

        Args:
            filename: Имя файла (если None, будет использован первый найденный файл)
            raw: для xlsx читать без шапок (header=None) — файл рулетки имеет
                нестандартную структуру (несколько таблиц на одном листе)

        Returns:
            DataFrame с загруженными данными
        """
        if filename is None:
            # Если имя файла не указано, используем первый найденный файл.
            # Отфильтровываем подкаталоги (raw/, processed/) и незнакомые типы.
            files = [
                f for f in os.listdir(self.data_path)
                if os.path.isfile(os.path.join(self.data_path, f))
            ]
            supported = [f for f in files if f.lower().endswith(('.csv', '.xls', '.xlsx', '.json'))]
            if supported:
                files = supported
            if not files:
                raise FileNotFoundError(f"В директории {self.data_path} не найдено файлов")
            filename = files[0]
        
        file_path = os.path.join(self.data_path, filename)
        
        # Определение формата файла по расширению
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            return pd.read_excel(file_path, header=None) if raw else pd.read_excel(file_path)
        elif ext.lower() == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
    
    def save_processed_data(self, data, filename):
        """
        Сохранение обработанных данных.
        
        Args:
data: DataFrame с обработанными данными
            filename: Имя файла для сохранения
        """
        processed_dir = os.path.join(self.data_path, 'processed')
        
        # Создаем директорию, если она не существует
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        file_path = os.path.join(processed_dir, filename)
        
        # Определение формата файла по расширению
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.csv':
            data.to_csv(file_path, index=False)
        elif ext.lower() in ['.xls', '.xlsx']:
            data.to_excel(file_path, index=False)
        elif ext.lower() == '.json':
            data.to_json(file_path, orient='records')
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        
        print(f"Обработанные данные сохранены в {file_path}")