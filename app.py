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

# Fungsi untuk memuat gambar dari folder
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

# Fungsi untuk memuat dan menampilkan gambar dari unggahan pengguna
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

# Konversi gambar ke array numpy
def convert_image_to_array(image):
    return np.array(image)

# Ubah ukuran gambar
def resize_image(image_array, size=(128, 128)):
    image_pil = Image.fromarray(image_array)
    resized_image_pil = image_pil.resize(size)
    return np.array(resized_image_pil)

# Normalisasi gambar
def normalize_image(image_array):
    return image_array / 255.0

# Ekstrak fitur (histogram warna dan tekstur)
def extract_features(image_array):
    # Fitur histogram warna
    color_hist_features = []
    for channel in range(3):
        histogram, _ = np.histogram(image_array[:, :, channel], bins=256, range=(0, 256))
        histogram = histogram.astype('float')
        histogram /= histogram.sum()
        color_hist_features.extend(histogram)
    
    # Rata-rata warna
    avg_color = np.mean(image_array, axis=(0, 1))
    avg_color_features = avg_color / 255.0
    
    # Konversi gambar grayscale ke uint8
    gray_image = color.rgb2gray(image_array)
    gray_image_uint8 = (gray_image * 255).astype('uint8')
    
    # Fitur Local Binary Pattern untuk tekstur
    lbp = feature.local_binary_pattern(gray_image_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()
    
    # Gabungkan fitur warna dan tekstur
    feature_vector = np.concatenate((color_hist_features, avg_color_features, lbp_hist))
    return feature_vector

# Visualisasi PCA
def visualize_clusters(X, labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title('Visualisasi Klaster menggunakan PCA')
    ax.set_xlabel('Komponen PCA 1')
    ax.set_ylabel('Komponen PCA 2')
    fig.colorbar(scatter, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# Menampilkan gambar dari setiap klaster dalam grid
def show_cluster_images(images, labels, filenames, cluster_num, num_images=5):
    cluster_indices = np.where(labels == cluster_num)[0]
    if len(cluster_indices) == 0:
        st.write(f"Tidak ada gambar dalam klaster {cluster_num}.")
        return
    selected_indices = np.random.choice(cluster_indices, min(len(cluster_indices), num_images), replace=False)
    st.write(f'Sample Gambar dari Klaster {cluster_num}')
    
    # Menentukan jumlah kolom berdasarkan jumlah gambar yang dipilih
    cols = st.columns(len(selected_indices))
    for i, idx in enumerate(selected_indices):
        with cols[i]:
            st.image(images[idx], caption=filenames[idx], use_column_width=True)

# Program Utama
def main():
    st.title("Aplikasi Klasterisasi Gambar")
    st.sidebar.header("Pengaturan")
    st.sidebar.write("Pilih jumlah klaster, muat gambar, dan unggah citra tambahan untuk dikelompokkan menggunakan KMeans")
    
    # Langkah 1: Pilih jumlah klaster
    k = st.sidebar.slider('Pilih jumlah klaster (k)', 2, 10, 4, key='k_slider')
    
    # Langkah 2: Muat gambar dari folder
    st.sidebar.subheader("Muat Gambar dari Folder")
    folder_path = st.sidebar.text_input("Masukkan path folder untuk memuat gambar:")
    if st.sidebar.button("Muat Gambar", key='load_images'):
        if folder_path:
            images, filenames = load_images_from_folder(folder_path)
            if len(images) == 0:
                st.error("Tidak ada gambar yang ditemukan di folder yang ditentukan.")
                return
            st.success(f"{len(images)} gambar berhasil dimuat.")
            
            # Simpan gambar dan nama file dalam session state
            st.session_state['images'] = images
            st.session_state['filenames'] = filenames
            st.session_state['folder_loaded'] = True
        else:
            st.error("Silakan masukkan path folder yang valid.")
    
    # Tombol untuk Memulai Klasterisasi
    if st.sidebar.button("Mulai Klasterisasi", key='start_clustering'):
        if 'images' not in st.session_state or 'filenames' not in st.session_state:
            st.error("Silakan muat gambar dari folder terlebih dahulu sebelum memulai klasterisasi.")
        else:
            images = st.session_state['images']
            filenames = st.session_state['filenames']
            
            all_features = []
            for image in images:
                original_image_array = convert_image_to_array(image)
                resized_image_array = resize_image(original_image_array, size=(128, 128))
                normalized_image_array = normalize_image(resized_image_array)
    
                features = extract_features(normalized_image_array)
                all_features.append(features)
    
            X = np.array(all_features)
            st.write(f"Total fitur yang dikumpulkan: {X.shape}")
    
            # Skala Fitur
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state['scaler'] = scaler  # Simpan scaler ke session state
    
            # Klasterisasi KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            st.session_state['kmeans'] = kmeans  # Simpan model KMeans ke session state
            st.session_state['labels'] = labels  # Simpan label ke session state
    
            st.write(f"Klasterisasi KMeans diterapkan dengan k={k}.")
    
            # Evaluasi Silhouette Score
            sil_score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette Score untuk k={k}: {sil_score:.4f}")
    
            # Visualisasi PCA
            visualize_clusters(X_scaled, labels)
    
            # Menampilkan gambar dari setiap klaster
            st.subheader("Gambar per Klaster")
            for cluster in range(k):
                show_cluster_images(images, labels, filenames, cluster_num=cluster, num_images=5)
    
    # Langkah 3: Unggah gambar tambahan
    st.sidebar.subheader("Unggah Citra Tambahan")
    uploaded_files = st.sidebar.file_uploader("Unggah gambar Anda sendiri", accept_multiple_files=True, type=["png", "jpg", "jpeg", "bmp", "tiff"])
    
    if uploaded_files:
        if 'kmeans' not in st.session_state or 'scaler' not in st.session_state:
            st.warning("Silakan muat dan klasterisasi gambar dari folder terlebih dahulu untuk melakukan klasterisasi pada gambar yang diunggah.")
        else:
            images_uploaded, filenames_uploaded = load_uploaded_images(uploaded_files)
    
            all_features_uploaded = []
            for image in images_uploaded:
                original_image_array = convert_image_to_array(image)
                resized_image_array = resize_image(original_image_array, size=(128, 128))
                normalized_image_array = normalize_image(resized_image_array)
    
                features = extract_features(normalized_image_array)
                all_features_uploaded.append(features)
    
            X_uploaded = np.array(all_features_uploaded)
            st.write(f"Total fitur yang dikumpulkan dari gambar yang diunggah: {X_uploaded.shape}")
    
            # Skala Fitur untuk gambar yang diunggah
            scaler = st.session_state['scaler']
            features_scaled_uploaded = scaler.transform(X_uploaded) 
    
            # Prediksi klaster untuk gambar yang diunggah
            kmeans = st.session_state['kmeans']
            labels_uploaded = kmeans.predict(features_scaled_uploaded)
    
            # Menampilkan hasil untuk gambar yang diunggah
            st.subheader("Hasil Klasterisasi untuk Gambar yang Diunggah")
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Gambar '{uploaded_file.name}' termasuk dalam klaster: {labels_uploaded[i]}")
    
            # Menampilkan gambar dari file yang diunggah per klaster
            st.subheader("Gambar Unggahan per Klaster")
            for cluster in range(k):
                show_cluster_images(images_uploaded, labels_uploaded, [file.name for file in uploaded_files], cluster_num=cluster)
    
    # Menampilkan gambar yang diunggah di main area (opsional)
    # Uncomment jika ingin menampilkan gambar di main area
    # if 'images' in st.session_state and 'labels' in st.session_state:
    #     images = st.session_state['images']
    #     filenames = st.session_state['filenames']
    #     labels = st.session_state['labels']
    #     st.subheader("Gambar per Klaster")
    #     for cluster in range(k):
    #         show_cluster_images(images, labels, filenames, cluster_num=cluster, num_images=5)

if __name__ == "__main__":
    main()
