import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    # Load and preprocess data
    features_df = pd.read_csv("features.csv")
    labels_df = pd.read_excel("train/classif.xlsx")

    # Remove image 154 as it's missing its mask
    labels_df = labels_df[labels_df["ID"] != 154]
    features_df["ID"] = features_df["img_name"].str.extract(r'(\d+)').astype(int)

    # Merge features with classification data
    merged_df = pd.merge(features_df, labels_df, on="ID")

    # Filter out classes with less than 5 examples
    value_counts = merged_df['bug type'].value_counts()
    classes_to_keep = value_counts[value_counts >= 5].index
    filtered_df = merged_df[merged_df['bug type'].isin(classes_to_keep)]

    # Prepare features and target
    X = filtered_df.select_dtypes(include=[np.number]).drop(columns=["ID"])
    y = filtered_df["bug type"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()
        
        return accuracy
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")
        return None

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()
    
    print("Number of examples:", X_train_scaled.shape[0] + X_test_scaled.shape[0])
    print("Number of features:", X_train_scaled.shape[1])
    print("\nClass distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Dictionary to store accuracies
    accuracies = {}
    
    # 1. K-Nearest Neighbors
    print("\n=== K-Nearest Neighbors ===")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_accuracy = evaluate_model(knn, X_train_scaled, X_test_scaled, y_train, y_test, "KNN")
    if knn_accuracy is not None:
        accuracies["KNN"] = knn_accuracy
    
    # 2. Logistic Regression
    print("\n=== Logistic Regression ===")
    log_reg = LogisticRegression(max_iter=1000, solver='saga')
    log_reg_accuracy = evaluate_model(log_reg, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic_Regression")
    if log_reg_accuracy is not None:
        accuracies["Logistic Regression"] = log_reg_accuracy
    
    # 3. Random Forest
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_accuracy = evaluate_model(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random_Forest")
    if rf_accuracy is not None:
        accuracies["Random Forest"] = rf_accuracy
    
    # Compare models if we have any results
    if accuracies:
        print("\n=== Model Comparison ===")
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies.values()):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

if __name__ == "__main__":
    main()
