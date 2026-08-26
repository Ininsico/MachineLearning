class Reporter:
    @staticmethod
    def display_model_evaluation(metrics, title="MODEL PERFORMANCE"):
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        print(f"\n--- {title} ---")
        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"MAE:      {metrics['mae']:.2f}")
        print(f"Accuracy: {metrics['accuracy']:.1f}%")
        print("-" * (len(title) + 8) + "\n")
