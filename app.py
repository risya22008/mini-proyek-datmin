import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from skimage import feature, color
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Function to load images from a folder
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
                st.error(f"Failed to read {filename}: {e}")
    return images, filenames

# Function to load and display images from user upload
def load_uploaded_images(uploaded_files):
    images = []
    filenames = []
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            images.append(img)
            filenames.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
    return images, filenames

# Convert image to numpy array
def convert_image_to_array(image):
    return np.array(image)

# Resize image
def resize_image(image_array, size=(128, 128)):
    image_pil = Image.fromarray(image_array)
    resized_image_pil = image_pil.resize(size)
    return np.array(resized_image_pil)

# Normalize image
def normalize_image(image_array):
    return image_array / 255.0

# Extract features (color histogram and texture)
def extract_features(image_array):
    # Color histogram features
    color_hist_features = []
    for channel in range(3):
        histogram, _ = np.histogram(image_array[:, :, channel], bins=256, range=(0, 256))
        histogram = histogram.astype('float')
        histogram /= histogram.sum()
        color_hist_features.extend(histogram)
    
    # Color averages
    avg_color = np.mean(image_array, axis=(0, 1))
    avg_color_features = avg_color / 255.0
    
    # Convert grayscale image to uint8
    gray_image = color.rgb2gray(image_array)
    gray_image_uint8 = (gray_image * 255).astype('uint8')
    
    # Local Binary Pattern features for texture
    lbp = feature.local_binary_pattern(gray_image_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()
    
    # Combine color and texture features
    feature_vector = np.concatenate((color_hist_features, avg_color_features, lbp_hist))
    return feature_vector

# Main Program
def main():
    st.title("Image Clustering Application")
    st.sidebar.header("Settings")
    st.sidebar.write("Upload images to cluster them using KMeans")

    # Load images from a specified folder
    folder_path = st.sidebar.text_input("Enter folder path to load images:")
    if st.sidebar.button("Load Images"):
        if folder_path:
            images, filenames = load_images_from_folder(folder_path)
            if len(images) == 0:
                st.error("No images found in the specified folder.")
                return
            st.success(f"{len(images)} images loaded.")

            all_features = []
            for image in images:
                original_image_array = convert_image_to_array(image)
                resized_image_array = resize_image(original_image_array, size=(128, 128))
                normalized_image_array = normalize_image(resized_image_array)

                features = extract_features(normalized_image_array)
                all_features.append(features)

            X = np.array(all_features)
            st.write(f"Total features collected: {X.shape}")

            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # KMeans Clustering
            k = st.sidebar.slider('Select number of clusters (k)', 2, 10, 4)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            st.write(f"KMeans clustering applied with k={k}.")

            # Silhouette Score Evaluation
            sil_score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette Score for k={k}: {sil_score:.4f}")

            # PCA Visualization
            def visualize_clusters(X, labels):
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)

                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
                plt.title('Cluster Visualization using PCA')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.colorbar(scatter)
                st.pyplot(plt)

            visualize_clusters(X_scaled, labels)

            # Show images from each cluster in a grid
            def show_cluster_images(images, labels, filenames, cluster_num, num_images=5):
                cluster_indices = np.where(labels == cluster_num)[0]
                if len(cluster_indices) == 0:
                    st.write(f"No images in cluster {cluster_num}.")
                    return
                selected_indices = np.random.choice(cluster_indices, min(len(cluster_indices), num_images), replace=False)
                st.write(f'Sample Images from Cluster {cluster_num}')
                cols = st.columns(num_images)
                for i, idx in enumerate(selected_indices):
                    with cols[i]:
                        st.image(images[idx], caption=filenames[idx], use_column_width=True)

            # Show images from each cluster
            for cluster in range(k):
                show_cluster_images(images, labels, filenames, cluster_num=cluster, num_images=5)

        else:
            st.error("Please enter a valid folder path.")

    # Upload images from user
    uploaded_files = st.file_uploader("Upload your own images", accept_multiple_files=True, type=["png", "jpg", "jpeg", "bmp", "tiff"])
    
    if uploaded_files:
        images_uploaded, filenames_uploaded = load_uploaded_images(uploaded_files)

        all_features_uploaded = []
        for image in images_uploaded:
            original_image_array = convert_image_to_array(image)
            resized_image_array = resize_image(original_image_array, size=(128, 128))
            normalized_image_array = normalize_image(resized_image_array)

            features = extract_features(normalized_image_array)
            all_features_uploaded.append(features)

        X_uploaded = np.array(all_features_uploaded)
        st.write(f"Total features collected from uploaded images: {X_uploaded.shape}")

        # Feature Scaling for uploaded images
        features_scaled_uploaded = scaler.transform(X_uploaded) 

        # Predict clusters for uploaded images
        labels_uploaded = kmeans.predict(features_scaled_uploaded)

        # Show results for uploaded images
        st.subheader("Clustering Results for Uploaded Images")
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"The uploaded image '{uploaded_file.name}' belongs to cluster: {labels_uploaded[i]}")

        # Display images from the uploaded files per cluster
        for cluster in range(k):
            st.subheader(f"Images from Cluster {cluster}")
            show_cluster_images(images_uploaded, labels_uploaded, [uploaded_file.name for uploaded_file in uploaded_files], cluster_num=cluster)

if __name__ == "__main__":
    main()
