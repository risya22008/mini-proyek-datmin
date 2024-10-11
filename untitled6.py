import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                filenames.append(filename)
                print(f"Loaded image: {filename}")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    return images, filenames

# 2. Convert image to numpy array
def convert_image_to_array(image):
    return np.array(image)

# 3. Resize image
def resize_image(image_array, size=(128, 128)):
    image_pil = Image.fromarray(image_array)
    resized_image_pil = image_pil.resize(size)
    return np.array(resized_image_pil)

# 4. Normalize image
def normalize_image(image_array):
    return image_array / 255.0

# 5. Extract features
def extract_features(image_array):
    # Color histogram features
    color_hist_features = []
    for channel in range(3):
        histogram, _ = np.histogram(image_array[:, :, channel], bins=256, range=(0, 256))
        histogram = histogram.astype('float')
        histogram /= histogram.sum()
        color_hist_features.extend(histogram)
    color_hist_features = np.array(color_hist_features)
    
    # Color averages
    avg_color = np.mean(image_array, axis=(0, 1))  # Average RGB values
    avg_color_features = avg_color / 255.0  # Normalize color values

    # Convert grayscale image to uint8
    gray_image = color.rgb2gray(image_array)
    gray_image_uint8 = (gray_image * 255).astype('uint8')  # Convert to uint8
    
    # Local Binary Pattern features for texture
    lbp = feature.local_binary_pattern(gray_image_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()
    
    # Combine color and texture features
    feature_vector = np.concatenate((color_hist_features, avg_color_features, lbp_hist))
    return feature_vector

# 6. Main Program
def main():
    folder_path = './Kmeans'  # Change to your image folder path
    images, filenames = load_images_from_folder(folder_path)

    if len(images) == 0:
        print("No images were loaded. Check the folder path.")
        return

    all_features = []
    for image in images:
        original_image_array = convert_image_to_array(image)
        resized_image_array = resize_image(original_image_array, size=(128, 128))
        normalized_image_array = normalize_image(resized_image_array)

        features = extract_features(normalized_image_array)
        all_features.append(features)

    X = np.array(all_features)
    print(f"Total features collected: {X.shape}")

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Clustering with 4 clusters
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    print(f"KMeans clustering applied with k={k}.")

    # Silhouette Score Evaluation
    sil_score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score for k={k}: {sil_score:.4f}")

    # PCA Visualization
    def visualize_clusters(X, labels):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.title('Cluster Visualization using PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.colorbar(scatter)
        plt.show()

    visualize_clusters(X_scaled, labels)

    # Show images from each cluster
    def show_cluster_images(images, labels, filenames, cluster_num, num_images=5):
        cluster_indices = np.where(labels == cluster_num)[0]
        if len(cluster_indices) == 0:
            print(f"No images in cluster {cluster_num}.")
            return
        selected_indices = np.random.choice(cluster_indices, min(len(cluster_indices), num_images), replace=False)
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(selected_indices):
            plt.subplot(1, num_images, i + 1)
            img = images[idx].astype('uint8')
            plt.imshow(img)
            plt.title(f"{filenames[idx]}")
            plt.axis('off')
        plt.suptitle(f'Sample Images from Cluster {cluster_num}')
        plt.show()

    # Show images from each cluster
    for cluster in range(k):
        show_cluster_images(np.array(images), labels, filenames, cluster_num=cluster, num_images=5)

if __name__ == "__main__":
    main()
