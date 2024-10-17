import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
from scipy import ndimage

# Set page configuration
st.set_page_config(page_title="Image Segmentation App", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #1E201E;
        color: #ECDFCC;
    }
    .stButton>button {
        width: 100%;
        background-color: #697565;
        color: #ECDFCC;
    }
    .stProgress .st-bo {
        background-color: #ECDFCC;
    }
    .stSlider>div>div>div>div {
        background-color: #ECDFCC;
    }
    .stSlider>div>div>div:before {
        background-color: #ECDFCC; /* Warna krem untuk thumb (pegangan) slider */
    }
    .stSlider>div>div>div {
        color: #ECDFCC; /* Ganti warna teks slider */
    }
    .stSelectbox>div>div {
        background-color: #697565;
        color: #ECDFCC;
    }
    .stMarkdown {
        color: #697565;
    }
    .stSidebar {
        background-color: #ECDFCC;
    }
    .stExpander {
        background-color: #3C3D37;
    }
    .stFileUploader {
        background-color: #ECDFCC; 
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_image(image, size=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

def plot_segmented_image(original_img, segmented_img, num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(segmented_img, cmap="viridis")
    ax2.set_title(f"Segmented Image: {num_clusters} Clusters")
    ax2.axis('off')
    
    for i in range(num_clusters):
        mask = segmented_img == i
        center = ndimage.measurements.center_of_mass(mask)
        ax2.text(center[1], center[0], str(i+1), 
                 color='white', fontsize=12, ha='center', va='center')
    
    return fig

def process_image(img, max_clusters):
    preprocessed_img = preprocess_image(np.array(img))
    pixel_values = preprocessed_img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    best_score = -1
    best_num_clusters = 0
    best_segmented_image = None
    results = []

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(pixel_values)
        segmented_img = labels.reshape(preprocessed_img.shape[:2])
        
        score = silhouette_score(pixel_values, labels)
        results.append((num_clusters, score, segmented_img))

        if score > best_score:
            best_score = score
            best_num_clusters = num_clusters
            best_segmented_image = segmented_img

    return best_score, best_num_clusters, best_segmented_image, preprocessed_img, results

# Streamlit App
st.title("🖼️ Multi-Image Segmentation using K-means Clustering 🖼️")

st.sidebar.header("Settings")
max_clusters = st.sidebar.slider("Max number of clusters", 2, 10, 5)
uploaded_files = st.sidebar.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Run K-means Clustering", key="run_button"):
        for idx, uploaded_file in enumerate(uploaded_files):
            st.header(f"📊 Processing image: {uploaded_file.name}")
            
            img = Image.open(uploaded_file)
            
            with st.spinner("Clustering in progress..."):
                best_score, best_num_clusters, best_segmented_image, preprocessed_img, results = process_image(img, max_clusters)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(img, use_column_width=True)
            
            with col2:
                st.subheader("Best Segmentation Result")
                fig = plot_segmented_image(preprocessed_img, best_segmented_image, best_num_clusters)
                st.pyplot(fig)

            st.success(f"Best Silhouette Score: {best_score:.4f} (with {best_num_clusters} clusters)")

            # Display all results in an expander
            with st.expander("See all segmentation results"):
                for num_clusters, score, segmented_img in results:
                    st.write(f"Clusters: {num_clusters}, Silhouette Score: {score:.4f}")
                    fig = plot_segmented_image(preprocessed_img, segmented_img, num_clusters)
                    st.pyplot(fig)

            # Progress bar
            st.progress((idx + 1) / len(uploaded_files))
            
            if idx < len(uploaded_files) - 1:
                st.markdown("---")
else:
    st.info("Please upload one or more images to begin segmentation.")

st.sidebar.markdown("---")