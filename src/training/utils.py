import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Linux
import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    """
    Grafica la pérdida y precisión del entrenamiento y validación.

    Args:
        history (dict): Diccionario con las métricas de entrenamiento y validación.
            Debe contener las claves: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path (str): Ruta donde guardar la figura. Si es None, no se guarda.
    """
    plt.figure(figsize=(12, 5))

    # Gráfica de pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Gráfica de precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Val Accuracy', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.close()
