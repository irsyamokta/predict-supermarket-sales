import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
def load_data():
    data = pd.read_csv("dataset/supermarket_sales - Sheet1.csv")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Month'] = data['Date'].dt.month
    return data

data = load_data()

# Streamlit UI
st.title("Prediksi Penjualan Supermarket dengan KNN & SVM")

# Menampilkan dataset awal
st.subheader("Dataset Awal")
st.write(data.head())

# Visualisasi distribusi produk
st.subheader("Distribusi Produk")
fig, ax = plt.subplots(figsize=(8, 6))
data['Product line'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Distribusi Produk")
ax.set_xlabel("Product Line")
ax.set_ylabel("Jumlah Penjualan")
plt.xticks(rotation=45)
st.pyplot(fig)

# Pilihan model
st.subheader("Pilih Model untuk Menampilkan Hasil")
model_option = st.selectbox("Model",["KNN", "SVM"])

# Membaca hasil prediksi dan rekomendasi dari file CSV berdasarkan pilihan model
if model_option == "KNN":
    predictions_df = pd.read_csv('./result/knn/predictions_and_recommendations_knn.csv')
    metrics_df = pd.read_csv('./result/knn/model_metrics_knn.csv')  # Ganti dengan path yang sesuai
elif model_option == "SVM":
    predictions_df = pd.read_csv('./result/svm/predictions_and_recommendations_svm.csv')
    metrics_df = pd.read_csv('./result/svm/model_metrics_svm.csv')  # Ganti dengan path yang sesuai

# Menampilkan RMSE dan Akurasi
st.subheader("Metrik Evaluasi Model")
st.write(f"📉 **RMSE: {metrics_df['RMSE'][0]:.4f}**")
st.write(f"✅ **Akurasi: {metrics_df['Accuracy (%)'][0]:.2f}%**")

# Visualisasi Prediksi
st.subheader("Prediksi Penjualan dan Rekomendasi Restock")
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Mengatur lebar batang
width = 0.4  # Lebar batang

# Posisi x untuk batang Predicted Quantity
x = np.arange(len(predictions_df['Product line']))

# Grafik Predicted Quantity
bars1 = ax1.bar(x - width/2, predictions_df['Predicted_Quantity'], 
                 color='blue', label='Predicted Quantity', width=width)

# Grafik Recommended Quantity
bars2 = ax1.bar(x + width/2, predictions_df['Recommended_Quantity'], 
                 color='orange', alpha=0.5, label='Recommended Quantity', width=width)

ax1.set_title('Prediksi dan Rekomendasi Restock')
ax1.set_xlabel('Product Line')
ax1.set_ylabel('Quantity')
ax1.set_xticks(x)
ax1.set_xticklabels(predictions_df['Product line'], rotation=45)
ax1.legend()

# Menambahkan label pada setiap batang diagram (jumlah prediksi)
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # Offset
                 textcoords="offset points",
                 ha='center', va='bottom')

# Menambahkan label pada setiap batang diagram (jumlah rekomendasi)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # Offset
                 textcoords="offset points",
                 ha='center', va='bottom')

st.pyplot(fig1)