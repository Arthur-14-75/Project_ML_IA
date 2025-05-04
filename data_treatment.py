# Feature extraction and data treatment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import measure
import os
import re

# bibliothèque pour la partie projection des données
from sklearn.manifold import TSNE
import umap

# bibliothèque pour la partie clustering
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


def extract_numbers(filename):
    # Use regex to find all numbers in the filename
    match = re.findall(r'\d+', filename)
    if match:
        return int(match[0])
    else:
        return -1

img_path = 'train/images/'
mask_path = 'train/masks/'



images = sorted([f for f in os.listdir(img_path) if f.endswith('.JPG')],
                key=lambda f: extract_numbers(f)) 
masks = sorted([f for f in os.listdir(mask_path) if f.endswith('.tif')],
               key=lambda f: extract_numbers(f))
#print(f"Sorted images: {images[:10]}")  # Affiche les 10 premières images triées
#print(f"Sorted masks: {masks[:10]}")

image_numbers = sorted([extract_numbers(f) for f in images])
mask_numbers = sorted([extract_numbers(f) for f in masks])

missing_mask_numbers = set(image_numbers) - set(mask_numbers)
missing_image_numbers = set(mask_numbers) - set(image_numbers)

if missing_mask_numbers:
    print("Image(s) sans masque correspondant :", missing_mask_numbers)
    print("Nom(s) de fichier image concerné(s) :")
    for img in images:
        if extract_numbers(img) in missing_mask_numbers:
            print("  -", img)

if missing_image_numbers:
    print("Masque(s) sans image correspondante :", missing_image_numbers)
    print("Nom(s) de fichier masque concerné(s) :")
    for mask in masks:
        if extract_numbers(mask) in missing_image_numbers:
            print("  -", mask)

if 154 in image_numbers:
    image_numbers.remove(154)
    # Supprimer l'image correspondante de images
    images = [img for img in images if extract_numbers(img) != 154]
    # Supprimer aussi le masque correspondant si nécessaire
    masks = [mask for mask in masks if extract_numbers(mask) != 154]

print("Image numbers:", len(image_numbers))
print("Mask numbers:", len(mask_numbers))
features = []
if len(image_numbers) != len(mask_numbers):
    print(f"Warning: Number of images ({len(images)}) does not match number of masks ({len(masks)})")

min_length = min(len(images), len(masks))


for i in range (min_length):
    # Extract features from the image and mask
    #print(f"Processing image: {image}")
    #print(f"Processing mask: {mask}")

    img = cv2.imread(os.path.join(img_path, images[i]))
    mask_img = cv2.imread(os.path.join(mask_path, masks[i]), 0)


    if img is None:
        print(f"Error loading image: {images[i]}")
        continue  # Passer à l'image suivante

    if mask_img is None:
        print(f"Error loading mask: {mask[i]}")
        continue  # Passer à l'image suivante

    shappe_img = img.shape
    shappe_mask = mask_img.shape

    if shappe_img[0] != shappe_mask[0] or shappe_img[1] != shappe_mask[1]:
        mask = cv2.resize(mask_img, (shappe_img[1], shappe_img[0]), interpolation=cv2.INTER_NEAREST)

    ratio = mask_img.sum() / (shappe_mask[0] * shappe_mask[1])

    bug_Area = img[mask_img >0]
    if bug_Area.size == 0:  
        print(f"No bug area detected in mask: {mask}")
        continue

    r, g, b = bug_Area[:, 0], bug_Area[:, 1], bug_Area[:, 2]
    bug_Area_size= np.count_nonzero(mask_img)

    rgb_features = {"r_min": r.min(),"r_max": r.max(), "r_mean": r.mean(),"r_median":np.median(r), "r_std": r.std(),
                    "g_min": g.min(),"g_max": g.max(), "g_mean": g.mean(), "g_median":np.median(g),"g_std": g.std(),
                    "b_min": b.min(),"b_max": b.max(), "b_mean": b.mean(), "b_median":np.median(b),"b_std": b.std()}
    
    contours = measure.find_contours(mask_img, 0.5)
    valid_contours = [c for c in contours if len(c) >= 3]

    if valid_contours:
        contour = max(valid_contours, key=lambda c: cv2.contourArea(np.array(c, dtype=np.float32)))
        contour  = np.array(contour.astype(np.float32))

        contour_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        convex_ratio = contour_area / hull_area if hull_area > 0 else 0
        excentricity = contour_area / (np.pi * (bug_Area_size / np.pi) ** 2) if bug_Area_size > 0 else 0
    else:
        convex_ratio = 0
        exentricity = 0


    all_features = {
        "img_name": images[i],
        "mask_name": masks[i],
        "img_shape_0": shappe_img[0],
        "img_shape_1": shappe_img[1],
        "mask_shape_0": shappe_mask[0],
        "mask_shape_1": shappe_mask[1],
        "ratio": ratio,
        "convex_ratio": convex_ratio,
        "exentricity": excentricity
    }

    all_features.update(rgb_features)
    features.append(all_features)

features_df = pd.DataFrame(features)

features_df.to_csv('features.csv', index=False)
features_df.to_excel("features.xlsx", index=False)

# =========== Maintenant qu'on a extrait les features, on peut visualiser les données ==========

#Répartition des types d'insectes et des espèces

#print(os.listdir())

df_class = pd.read_excel("train/classif.xlsx")
print("df_class.shape:", df_class.shape)
df_class = df_class[df_class['ID'] != 154]
print("df_class.shape:", df_class.shape)

bug_type_count = df_class['bug type'].value_counts()
print("bug_type_count:", bug_type_count)

plt.figure(figsize=(10, 6))
bug_type_count.plot(kind='bar', color='skyblue')
plt.title("Répartition des types d'insectes")
plt.xlabel('Espèce')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

species_count = df_class['species'].value_counts()

plt.figure(figsize=(10, 6))
species_count.plot(kind='bar', color='salmon')
plt.title('Répartition des espèces')
plt.xlabel('Espèce')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Ici On effectue une projection PCA sur les features extraites

numeric_df = features_df.select_dtypes(include=[np.number])

def variances(df):
    variances = df.var()
    return variances

vars = variances(numeric_df)

def center_data(df):
    nb_col = numeric_df.shape[1]
    df = df.astype(float)
    for i in range (nb_col):
        df.iloc[:, i] = df.iloc[:, i] - df.iloc[:, i].mean()
    return df

centered_data = center_data(numeric_df)

def covariance_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    nb_col = numeric_df.shape[1]
    cov_matrix = np.zeros((nb_col, nb_col))
    for i in range(nb_col):
        for j in range(nb_col):
            cov_matrix[i][j] = df.iloc[:, i].cov(df.iloc[:, j])
    return cov_matrix

cov_matrix = covariance_matrix(centered_data)

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

def sort_eigenvalues(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_values = eigenvalues[sorted_indices]
    sorted_vectors = eigenvectors[:, sorted_indices]
    return sorted_values, sorted_vectors

sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues(eigenvalues, eigenvectors)

def select_principal_components(eigenvalues, eigenvectors, k):
    select_eigenvalues = eigenvalues[:k]
    select_eigenvectors = eigenvectors[:, :k]
    return select_eigenvalues, select_eigenvectors

selected_values, selected_vectors = select_principal_components(sorted_eigenvalues, sorted_eigenvectors, 2)

def project_data(df, select_eigenvectors):
    nb_col = df.shape[1]
    projected_data = np.zeros((df.shape[0], select_eigenvectors.shape[1]))
    for i in range(df.shape[0]):
        for j in range (select_eigenvectors.shape[1]):
            projected_data[i][j] = df.iloc[i,:].dot(select_eigenvectors[:,j])

    return projected_data

projected_data = project_data(centered_data, sorted_eigenvectors)
print("projected_data.shape:", projected_data.shape)
#df_class = df_class.reset_index(drop=True)


def plot_pca(projected_data, df_class):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=projected_data[:, 0], y=projected_data[:, 1], hue=df_class['bug type'], palette='Set1')
    plt.title('PCA Projection of Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Bug Type')
    plt.show()

plot_pca(projected_data, df_class)

# =========== Maintenant qu'on a réalisé le pca, on va explorer une autre méthode de projection ==========

# ========== On va utiliser t-SNE pour projeter les données dans un espace 2D ==========

tsne = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_projected = tsne.fit_transform(numeric_df)

plt.figure(figsize=(8,6))
plt.scatter(tsne_projected[:, 0], tsne_projected[:, 1], color='red', alpha=0.5)
plt.title("t-SNE projection of features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()

# ========== On va maintenant essayer d'utiliser UMAP pour projeter les données dans un espace 2D ==========

reducer = umap.UMAP(random_state=0)
umap_projected = reducer.fit_transform(numeric_df)

plt.figure(figsize=(8,6))
plt.scatter(umap_projected[:, 0], umap_projected[:, 1], color='green', alpha=0.5)
plt.title("UMAP projection of features")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.show()

# ========== Dans cette partie on va faire un clustering sur les données projetées ==========

# 1. KMeans Clustering

X, y = projected_data , df_class['bug type'].values

kmeans = KMeans(n_clusters =bug_type_count.shape[0], random_state=42)
clusters = kmeans.fit_predict(X)

X_with_clusters = X.copy()
X_with_clusters = np.column_stack((X_with_clusters, clusters))

X_train, X_test, y_train, y_test = train_test_split(X_with_clusters, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

accuracy = log_reg.score(X_test, y_test)
print(f"Accuracy of Logistic Regression on KMeans clusters: {accuracy:.2f}")

score = silhouette_score(X_with_clusters, clusters) # sert à mesurer la cohérence des clusters
print(f"Silhouette Score: {score:.2f}") # plus il est proche de 1, mieux c'est

# Pour les résultats : plus de chevauchement entre les clusters pour X = projected_data donc score faible
# Groupes bien séparés pour X = umap_projected donc score élevé

# 2. DBSCAN Clustering

X = umap_projected
dbscan = DBSCAN(eps = 0.5, min_samples = 5)
clusters = dbscan.fit_predict(X)

score = silhouette_score(X, clusters)
print(f"Silhouette Score: {score:.2f}") 

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Nombre de clusters détectés : {n_clusters}")