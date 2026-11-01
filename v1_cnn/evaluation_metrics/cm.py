import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_cnn_performance(model, X_val_3d, y_val_encoded, encoder):
    print("\n--- 1D-CNN Evaluation ---")

    # Generate network predictions (outputs probabilities).
    y_pred_probs = model.predict(X_val_3d)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Calculate Macro F1-Score (weights all classes equally).
    macro_f1 = f1_score(y_val_encoded, y_pred_classes, average='macro')
    print(f"Validation Macro F1-Score: {macro_f1:.4f}")

    # Print detailed metrics for every activity.
    print("\nClassification Report:")
    print(
        classification_report(
            y_val_encoded,
            y_pred_classes,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    # Plot the confusion matrix.
    cm = confusion_matrix(y_val_encoded, y_pred_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=encoder.classes_
    ).plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title('Confusion Matrix: 1D-CNN')
    plt.tight_layout()
    plt.show()

    return macro_f1
