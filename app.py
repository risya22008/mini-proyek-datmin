import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from skimage import feature, color, segmentation
from sklearn.preprocessing import StandardScaler
import cv2

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
                st.write(f"Loaded image: {filename}")
            except Exception as e:
                st.error(f"Failed to read {filename}: {e}")
    return images, filenames

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

def convert_image_to_array(image):
    return np.array(image)

def resize_image(image_array, size=(256, 256)):
    return cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)

def normalize_image(image_array):
    return image_array / 255.0

def segment_image(image_array, n_segments=100):
    return segmentation.slic(image_array, n_segments=n_segments, compactness=10)

def extract_segment_features(image_array, segments):
    features = []
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        segment = image_array[mask]
        
        # Color features
        color_mean = np.mean(segment, axis= 0)
        color_std = np.std(segment, axis=0)
        
        # Texture features (using grayscale image)
        gray_segment = color.rgb2gray(segment)
        
        # Reshape gray_segment to 2D if it's 1D
        if gray_segment.ndim == 1:
            side_length = int(np.sqrt(gray_segment.shape[0]))
            gray_segment = gray_segment[:side_length**2].reshape(side_length, side_length)
        
        # Ensure gray_segment is not empty
        if gray_segment.size > 0:
            lbp = feature.local_binary_pattern(gray_segment, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, range=(0, 10), density=True)
        else:
            lbp_hist = np.zeros(10)  # Use a zero histogram if the segment is empty
        
        # Combine features
        segment_features = np.concatenate([color_mean, color_std, lbp_hist])
        features.append(segment_features)
    
    return np.array(features)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def kmeans(X, n_clusters, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0  # Only one cluster or all noise points
    
    silhouette_values = []
    
    for i in range(len(X)):
        a = np.mean([euclidean_distance(X[i], X[j]) for j in range(len(X)) if labels[j] == labels[i] and i != j])
        b = min([np.mean([euclidean_distance(X[i], X[j]) for j in range(len(X)) if labels[j] == label]) 
                 for label in unique_labels if label != labels[i] and label != -1])
        
        silhouette_values.append((b - a) / max(a, b))
    
    return np.mean(silhouette_values)

def visualize_clustered_segments(image_array, segments, labels, show_text=True):
    """
    Visualizes an image with clustered segments, optionally showing text (labels) on each segment.
    """
    clustered_image = image_array.copy()  # Keep the original image colors
    unique_segments = np.unique(segments)

    for idx, segment_id in enumerate(unique_segments):
        mask = segments == segment_id
        y, x = np.where(mask)
        
        if len(y) > 0 and len(x) > 0:
            clustered_image[mask] = image_array[mask]  # Keep original colors
            
            # Compute the center of the segment
            center_y, center_x = int(np.mean(y)), int(np.mean(x))
            
            # Get the cluster label for the current segment
            label = labels[idx]
            
            # Optionally add the label as text
            if show_text:
                plt.text(center_x, center_y, str(label), color='red', fontsize=12, 
                         ha='center', va='center', fontweight='bold', backgroundcolor='white')

    return clustered_image


def visualize_all_clustered_segments(images, all_segments, labels_list, n_clusters_list):
    """
    Visualize original images and clustered versions in a grid with a row for each cluster setting.
    """
    n_images = len(images)
    n_rows = len(n_clusters_list) + 1  # Adding 1 for the original image row
    
    # Create a figure with appropriate size based on the number of images and clusters
    fig, axs = plt.subplots(n_rows, n_images, figsize=(5 * n_images, 5 * n_rows))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Clustering Results", fontsize=16)
    
    # Display original images at the top row
    for i, image in enumerate(images):
        axs[0, i].imshow(image)
        axs[0, i].axis('off')
    axs[0, 0].set_title("Original Image", fontsize=14)
    
    # Perform clustering visualization for each n_clusters
    for row, n_clusters in enumerate(n_clusters_list, start=1):
        labels = labels_list[row - 1]
        start_idx = 0  # Reset the starting index for each row (n_clusters)

        # Process each image individually
        for col, (image, segments) in enumerate(zip(images, all_segments)):
            image_array = np.array(image)
            n_segments = len(np.unique(segments))
            
            # Get the labels corresponding to the current image's segments
            image_labels = labels[start_idx:start_idx + n_segments]
            start_idx += n_segments  # Update start_idx for the next image
            
            # Visualize the image with its clusters and labels
            clustered_image = visualize_clustered_segments(image_array, segments, image_labels, show_text=True)
            axs[row, col].imshow(clustered_image)
            axs[row, col].axis('off')
        
        axs[row, 0].set_title(f"{n_clusters} Clusters", fontsize=14)
    
    return fig

def main():
    st.title("Segment-Level Image Clustering Application")
    st.sidebar.header("Settings")
    
    # Step 1: Load images
    st.sidebar.subheader("Load Images")
    folder_path = st.sidebar.text_input("Enter folder path to load images:")
    uploaded_files = st.sidebar.file_uploader("Upload your own images", accept_multiple_files=True, type=["png", "jpg", "jpeg", "bmp", "tiff"])
    
    if st.sidebar.button("Load Images"):
        images = []
        filenames = []
        if folder_path:
            folder_images, folder_filenames = load_images_from_folder(folder_path)
            images.extend(folder_images)
            filenames.extend(folder_filenames)
        if uploaded_files:
            uploaded_images, uploaded_filenames = load_uploaded_images(uploaded_files)
            images.extend(uploaded_images)
            filenames.extend(uploaded_filenames)
        
        if len(images) == 0:
            st.error("No images found or uploaded.")
            return
        
        st.success(f"{len(images)} images successfully loaded.")
        st.session_state['images'] = images
        st.session_state['filenames'] = filenames
    
    # Step 2: Preprocess and extract features
    if 'images' in st.session_state:
        st.sidebar.subheader("Preprocessing and Feature Extraction")
        n_segments = st.sidebar.slider("Number of segments per image", 3, 10, 3)
        
        if st.sidebar.button("Process Images"):
            all_features = []
            all_segments = []
            
            for image in st.session_state['images']:
                image_array = convert_image_to_array(image)
                resized_image = resize_image(image_array)
                normalized_image = normalize_image(resized_image)
                segments = segment_image(normalized_image, n_segments)
                features = extract_segment_features(normalized_image, segments)
                
                all_features.append(features)
                all_segments.append(segments)
            
            # Combine all features
            combined_features = np.vstack(all_features)
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(combined_features)
            
            st.session_state['all_features'] = all_features
            st.session_state['all_segments'] = all_segments
            st.session_state['combined_features'] = combined_features
            st.session_state['normalized_features'] = normalized_features
            
            st.success("Features extracted successfully.")
    
    # Step 3: Clustering
    if 'combined_features' in st.session_state:
        st.sidebar.subheader("Clustering Settings")
        n_clusters_list = st.sidebar.multiselect("Select numbers of clusters to try:", [2, 3, 4, 5, 6, 7], [2, 3])
        
        if st.sidebar.button("Perform Clustering"):
            labels_list = []
            silhouette_scores = []
            
            for n_clusters in n_clusters_list:
                labels, _ = kmeans(st.session_state['normalized_features'], n_clusters=n_clusters)
                labels_list.append(labels)
                
                score = silhouette_score(st.session_state['normalized_features'], labels)
                silhouette_scores.append(score)
            
            st.session_state['labels_list'] = labels_list
            st.session_state['n_clusters_list'] = n_clusters_list
            st.session_state['silhouette_scores'] = silhouette_scores
            
            st.success("Clustering performed successfully.")
    
    # Step 4: Visualization
    if 'labels_list' in st.session_state:
        st.subheader("Clustering Results")
        
        fig = visualize_all_clustered_segments(
            st.session_state['images'], 
            st.session_state['all_segments'], 
            st.session_state['labels_list'], 
            st.session_state['n_clusters_list']
        )
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()