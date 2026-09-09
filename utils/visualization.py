"""
Модуль для визуализации данных и результатов.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        """
        Визуализация процесса обучения.
        """
        plt.figure(figsize=(12, 4))
        
        # График потерь
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # График точности
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(y_true, y_pred):
        """
        Визуализация предсказаний.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_true)), y_true, label='Actual', alpha=0.5)
        plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.5)
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
