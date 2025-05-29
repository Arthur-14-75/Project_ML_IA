from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

features_df = pd.read_csv(r"ADAPT\features.csv")
labels_df = pd.read_excel(r"ADAPT\train\classif.xlsx")

labels_df = labels_df[labels_df["ID"] != 154]
features_df["ID"] = features_df["img_name"].str.extract(r'(\d+)').astype(int)


merged_df = pd.merge(features_df, labels_df, on="ID")
X = merged_df.select_dtypes(include=[np.number]).drop(columns=["ID"])
y = merged_df["bug type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("Nb exemples :", X.shape[0])
print("Nb featuress :", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print("Lancement recherche par grille")
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


print("\n Résultats du GridSearch :")
print("Meilleurs paramètres:", grid_search.best_params_)
print("Score entraînement:", grid_search.best_score_)
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n Les résultats sur ce jeu de test :")
print("Accuracy du test :", acc*100,"%")
print("Classification trouvée:")
print(classification_report(y_test, y_pred))

#===============================On Filtrer les classes trop rares========================

# on supprime les classes avec moins de 5 exemples
value_counts = merged_df['bug type'].value_counts()
classes_to_keep = value_counts[value_counts >= 5].index
filtered_df = merged_df[merged_df['bug type'].isin(classes_to_keep)]

# Redéfinir X et y avec les données filtrées
X = filtered_df.select_dtypes(include=[np.number]).drop(columns=["ID"])
y = filtered_df["bug type"]

#===================Matrice d eocnfusion======================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#matrice de confusion 
cm = confusion_matrix(y_test, y_pred, labels=grid_search.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.best_estimator_.classes_)

#on plot "
plt.figure(figsize=(8, 6))
disp.plot(cmap="Reds", values_format='d')
plt.title("Matrice de confusion - SVM")
plt.grid(False)
plt.show()
