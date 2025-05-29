import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import measure
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
import os
import re
import json

# bibliothèque pour la partie projection des données
from sklearn.manifold import TSNE
import umap

# bibliothèque pour la partie clustering
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

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


def negative_radius(center):
    xc, yc = center
    xc = int(xc)
    yc = int(yc)
    if (0 <= int(yc) < distance_map.shape[0]) and (0 <= int(xc) < distance_map.shape[1]):
        return -distance_map[yc, xc]
    else:
        return np.inf


def rotate_image(arr, theta_degree, xc, yc):
   
     # Convert the angle into radian
    theta_opencv = -theta_degree
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((xc, yc), theta_opencv, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def symetrize_image(xc, arr):
    h, w = arr.shape
    
    # Partie gauche jusqu'à xc
    left = arr[:, :xc]
    # Partie droite à partir de xc
    right = arr[:, xc:]
    # Flip horizontal de la droite
    right_flipped = np.fliplr(right)

    # On crée une image symétrique de même taille
    sym = np.zeros_like(arr)
    
    # On garde la partie gauche
    sym[:, :xc] = left
    
    # Largeur disponible à gauche de xc
    available_width = xc
    # Largeur de la partie à coller
    copy_width = min(available_width, right_flipped.shape[1])
    
    # On colle la version flipped à gauche de xc
    sym[:, xc - copy_width:xc] = right_flipped[:, :copy_width]

    return sym

def symetry_loss(theta, arr, xc, yc, mask):
    angle = theta[0] if isinstance(theta, (np.ndarray, list)) else theta
    
    # Rotation de l’image et du masque
    rotated = rotate_image(arr, angle, xc, yc)
    rotated_mask = rotate_image(mask.astype(float), angle, xc, yc)

    # Création de l’image symétrique
    sym_img = symetrize_image(xc, rotated)

    # Différence au carré, limitée par le masque tourné
    diff = (rotated.astype(float) - sym_img.astype(float)) ** 2
    masked_diff = diff * rotated_mask

    return np.mean(masked_diff)

# ================== Alternative ===========================

def compute_symmetry_score(img):
    h, w = img.shape
    left = img[:, :w//2]
    right = np.fliplr(img[:, w - w//2:])
    # Recadrer au cas où la largeur est impaire
    min_w = min(left.shape[1], right.shape[1])
    score = np.sum(np.abs(left[:, :min_w] - right[:, :min_w]))
    return score

def rotate_and_expand(image, angle, center=None, border=0):
    h, w = image.shape[:2]
    
    # Si aucun centre n'est donné, utiliser le centre de l'image
    if center is None:
        center = (w / 2, h / 2)

    # Matrice de rotation
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calcul des dimensions du nouveau canvas
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adapter la matrice de rotation pour le recentrage
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Appliquer la rotation avec expansion
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    # Nouveau centre de l’image retournée
    new_center = (new_w // 2, new_h // 2)
    
    return rotated, new_center


def find_best_symmetry_angle(gray, mask_bin, center, angle_range=(-30, 30), step=1):
    best_angle = 0
    best_score = float('inf')
    best_rotated_gray = gray.copy()
    best_rotated_mask = mask_bin.copy()
    best_center = center

    for angle in range(angle_range[0], angle_range[1] + 1, step):
        rotated_gray, center_new = rotate_and_expand(gray, angle, center)
        rotated_mask, _ = rotate_and_expand(mask_bin, angle, center)

        masked = cv2.bitwise_and(rotated_gray, rotated_gray, mask=(rotated_mask > 0).astype(np.uint8))
        score = compute_symmetry_score(masked)

        if score < best_score:
            best_score = score
            best_angle = angle
            best_rotated_gray = rotated_gray
            best_rotated_mask = rotated_mask
            best_center = center_new

    return best_rotated_gray, best_rotated_mask, best_angle, best_center



def draw_symetry_axes(rotated_img, theta, xc, yc, color=(0, 0, 255), thickness=2):

    h, w = rotated_img.shape[:2]
    img_with_axis = rotated_img.copy()

    # Convertir l'angle en radians
    theta_rad = np.deg2rad(theta)

    # Vecteur directeur de l’axe (perpendiculaire au plan de symétrie)
    dx = np.cos(theta_rad)
    dy = np.sin(theta_rad)

    # Calculer deux points éloignés dans chaque direction pour tracer une longue ligne
    length = max(h, w) * 2
    x1 = int(xc - length * dx)
    y1 = int(yc - length * dy)
    x2 = int(xc + length * dx)
    y2 = int(yc + length * dy)

    # Tracer la ligne
    cv2.line(img_with_axis, (x1, y1), (x2, y2), color, thickness)
    return img_with_axis



test_theta = 0.0

for i in range(min_length):
    img_file = os.path.join(img_path, images[i])
    mask_file = os.path.join(mask_path, masks[i])

    img = cv2.imread(img_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Image file {images[i]} not found or unreadable.")
        continue
    if mask is None:
        print(f"Mask file {masks[i]} not found or unreadable.")
        continue

    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:
        print(f"Image {images[i]} has no valid mask or only one connected component, skipping.")
        continue

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    xc0, yc0 = centroids[largest_label]
    cleaned_mask = (labels == largest_label).astype(np.uint8)

    bounds = [(0, cleaned_mask.shape[1]-1), (0, cleaned_mask.shape[0]-1)]

    x, y, w, h, _ = stats[largest_label]
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = cleaned_mask[y:y+h, x:x+w]

    distance_map = distance_transform_edt(cropped_mask)

    result = minimize(negative_radius, x0=[xc0 - x, yc0 - y], bounds=bounds, method='L-BFGS-B')

    if not result.success:
        print(f"Circle fitting failed for image {images[i]}.")
        continue

    best_x, best_y = result.x
    best_radius = -result.fun
    xc_best, yc_best = int(best_x), int(best_y)
    print(f"Image {images[i]}: best center = ({xc_best}, {yc_best}), best radius = {best_radius}")
    center = (int(best_x + x), int(best_y + y))

    cricle_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
    cv2.circle(cricle_mask, (xc_best, yc_best), int(best_radius), 1, thickness=-1)

    gray_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
 # partie optimisation de la symétrie
    res_sym = minimize(lambda theta: symetry_loss(theta, gray_crop, xc_best, yc_best, cricle_mask),
                       x0=[0.0], bounds=[(-45, 45)], method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})

    if not res_sym.success:
        print(f"Symmetry alignment failed for image {images[i]} : {res_sym.message}")
        cv2.imwrite(f"debug/mask_failed_{images[i]}", cricle_mask * 255)

    contours, _ = cv2.findContours(cricle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"No contours found for image {images[i]}.")
        continue
    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) < 5:
        print(f"Contour insuffisant pour ellipse dans {images[i]}")
        continue
    ellipse = cv2.fitEllipse(cnt)
    (center_ellipse, axes, angle) = ellipse

    sym_angle = angle - 90 if axes[0] > axes[1] else angle
    print(f"Image {images[i]} : ellipse angle = {angle:.2f} → sym_axis = {sym_angle:.2f}")
    
    gray_sym, mask_sym, theta_best, center_new = find_best_symmetry_angle(
    gray_crop,
    cricle_mask,
    center=(xc_best, yc_best),  # centre du cercle
    angle_range=(-90, 90),
    step=1
    )

    image_with_axis = draw_symetry_axes(gray_sym, theta_best, center_new[0], center_new[1])

    os.makedirs('cleaned/images_circle', exist_ok=True)
    os.makedirs('cleaned/images_aligned_with_axis', exist_ok=True)
    os.makedirs('cleaned/masks', exist_ok=True)

    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_with_box, contours, -1, (0, 255, 0), 2)

    img_with_circle = img_with_box.copy()
    cv2.circle(img_with_circle, center, int(best_radius), (255, 0, 0), 2)

   

    
    cv2.imwrite(f'cleaned/images_circle/{images[i]}', img_with_circle)
    cv2.imwrite(f'cleaned/images_aligned_with_axis/{images[i].replace(".JPG", ".png")}', image_with_axis)
    cv2.imwrite(f'cleaned/masks/{images[i].replace(".JPG", ".png")}', cropped_mask * 255)








    
