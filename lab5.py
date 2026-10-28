import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans, vq, whiten
from skimage.transform import resize

K_CLUSTERS = 2  # Кількість кластерів
NUM_COMPONENTS = 20  # Кількість головних компонент
GRID_STEPS = 10  # Сітка 10x10 для сегментації
ITERATIONS = 10  # Кількість ітерацій k-means

FONTS_DIR = r"C:\Pyton\laba4\a_thumbs"
IMAGE_PATH = r"C:\Image\nmu.jpg"


# Допоміжна функція PCA
def pca(X):
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        M = np.dot(X, X.T)
        e, EV = np.linalg.eigh(M)
        e = np.clip(e, 0.0, None)
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e[::-1] + 1e-12)
        S_safe = np.where(S > 0, S, 1.0)
        V = V / S_safe[:, np.newaxis]
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    return V, S, mean_X


print("--- Виконання Завдання 1 ---")
font_files = [
    os.path.join(FONTS_DIR, f)
    for f in os.listdir(FONTS_DIR)
    if f.lower().endswith((".jpg", ".png"))
]
font_files.sort()
font_files = font_files[:100]  # Вибірка перших 100 символів

sample_im = Image.open(font_files[0]).convert("L")
im_size = sample_im.size

immatrix = np.array(
    [np.array(Image.open(f).convert("L")).flatten() for f in font_files],
    dtype=np.float32,
)

# Зменшення простору ознак до 20 головних компонент
V, S, immean = pca(immatrix)
projected = np.array(
    [
        np.dot(V[:NUM_COMPONENTS], immatrix[i] - immean.flatten())
        for i in range(len(font_files))
    ]
)

# Нормалізація ознак (whiten) та кластеризація
features_fonts = whiten(projected)
centroids, _ = kmeans(features_fonts, K_CLUSTERS, iter=ITERATIONS)
code, _ = vq(features_fonts, centroids)

# Відображення 10 зображень кожного класу (2 класи = 2 рядки по 10 колонок)
fig, axes = plt.subplots(K_CLUSTERS, 10, figsize=(15, 4))

for k in range(K_CLUSTERS):
    indices = np.where(code == k)[0]
    for i in range(10):
        ax = axes[k, i]
        if i < len(indices):
            img_show = Image.open(font_files[indices[i]])
            ax.imshow(img_show, cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title(f"Клас {k + 1}", loc="left")
plt.suptitle(
    f"Завдання 1: Кластеризація шрифтів (k={K_CLUSTERS}, ознак={NUM_COMPONENTS})"
)
plt.tight_layout()
plt.show()


print("--- Виконання Завдання 2 ---")
im_raster = Image.open(IMAGE_PATH)
im_arr = np.array(im_raster)

dx = im_arr.shape[0] // GRID_STEPS
dy = im_arr.shape[1] // GRID_STEPS

# Обчислення середніх значень RGB для кожного з блоків 10x10
features_blocks = []
for x in range(GRID_STEPS):
    for y in range(GRID_STEPS):
        R = np.mean(im_arr[x * dx : (x + 1) * dx, y * dy : (y + 1) * dy, 0])
        G = np.mean(im_arr[x * dx : (x + 1) * dx, y * dy : (y + 1) * dy, 1])
        B = np.mean(im_arr[x * dx : (x + 1) * dx, y * dy : (y + 1) * dy, 2])
        features_blocks.append([R, G, B])

features_blocks = np.array(features_blocks, dtype=np.float32)

centroids_img, _ = kmeans(features_blocks, K_CLUSTERS, iter=ITERATIONS)
code_img, _ = vq(features_blocks, centroids_img)

# Відновлення просторової картини кластерів під розмір оригінального зображення
codeim = code_img.reshape(GRID_STEPS, GRID_STEPS)
codeim_resized = resize(
    codeim,
    im_arr.shape[:2],
    order=0,
    preserve_range=True,
    anti_aliasing=False,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(im_arr)
axes[0].set_title("Вихідне зображення")
axes[0].axis("off")

axes[1].imshow(codeim_resized, cmap="viridis")
axes[1].set_title(f"Сегментація (k={K_CLUSTERS}, сітка {GRID_STEPS}x{GRID_STEPS})")
axes[1].axis("off")
plt.tight_layout()
plt.show()

print("--- Виконання Завдання 3 ---")
n_samples = len(projected)

# 1. Побудова матриці відстаней та матриці подібності
diff = projected[:, np.newaxis, :] - projected[np.newaxis, :, :]
dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

# Масштабування сигми через медіану відстаней
sigma = np.median(dist_matrix)
if sigma == 0:
    sigma = 1.0
S_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
np.fill_diagonal(S_matrix, 0.0)  # обнуляємо діагональ

# 2. Розрахунок матриці переходу P = D^(-1) * S
row_sums = S_matrix.sum(axis=1)
row_sums[row_sums == 0] = 1e-12
D_inv = np.diag(1.0 / row_sums)
P_matrix = np.dot(D_inv, S_matrix)

# 3. Обчислення власних векторів
evals, evecs = np.linalg.eig(P_matrix)
evals = np.real(evals)
evecs = np.real(evecs)

# Сортування за спаданням
idx = np.argsort(evals)[::-1]

# ПРОПУСКАЄМО перший тривіальний константний вектор (індекс 0)
# Беремо вектори від індексу 1 до K_CLUSTERS + 1
spectral_features = evecs[:, idx[1 : K_CLUSTERS + 1]]

# Нормалізація ознак
spectral_features = whiten(spectral_features)

# Кластеризація k-means
spectral_centroids, _ = kmeans(spectral_features, K_CLUSTERS, iter=20)
spectral_code, _ = vq(spectral_features, spectral_centroids)

# 4. Відображення результатів із правильними пропорціями
fig, axes = plt.subplots(K_CLUSTERS, 10, figsize=(14, 5))

for k in range(K_CLUSTERS):
    indices = np.where(spectral_code == k)[0]
    for i in range(10):
        ax = axes[k, i]
        if i < len(indices):
            img_show = Image.open(font_files[indices[i]]).convert("L")
            ax.imshow(img_show, cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title(f"Спектр. клас {k + 1}", loc="left", fontsize=10)

plt.suptitle(
    f"Завдання 3: Спектральна кластеризація матрицею D^(-1)S (k={K_CLUSTERS})",
    fontsize=12,
)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.show()