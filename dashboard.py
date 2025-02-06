import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import os

def load_data(file):
    data = pd.read_csv(file)
    if data.empty:
        st.error("Dataset yang diunggah kosong. Harap unggah dataset yang valid.")
        st.stop()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Month'] = data['Date'].dt.month
    return data

def preprocess_data(data):
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        st.warning(f"Dataset memiliki {total_missing} missing values")
    
    data = data.copy()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].mean(), inplace=True)

    data.drop_duplicates(inplace=True)
    grouped_data = data.groupby(['Product line', 'Month']).agg(
        Total_Quantity=('Quantity', 'sum'),
        Average_Unit_Price=('Unit price', 'mean')
    ).reset_index()
    return grouped_data

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = (1 - rmse / np.mean(y_true)) * 100
    return rmse, accuracy

st.title("Prediksi Penjualan Supermarket")
st.sidebar.header("Upload Dataset Anda")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Unduh template CSV
st.sidebar.subheader("Unduh Template CSV")
template_path = "dataset/template_dataset.csv"

if os.path.exists(template_path):
    with open(template_path, "rb") as file:
        st.sidebar.download_button(
            label="Download Template CSV",
            data=file,
            file_name="template_dataset.csv",
            mime="text/csv"
        )
else:
    st.sidebar.error("File template_dataset.csv tidak ditemukan.")

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.warning("Silakan unggah file CSV untuk melanjutkan.")
    st.stop()

df = preprocess_data(data)

if df.empty:
    st.error("Dataset tidak memiliki cukup data setelah preprocessing.")
    st.stop()

X = df[['Month', 'Average_Unit_Price']]
y = df['Total_Quantity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if len(y) < 5:
    st.error("Data tidak cukup untuk melatih model KNN dengan n_neighbors=5. Kurangi nilai n_neighbors.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

n_neighbors = min(5, len(X_train))
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

rmse_knn, accuracy_knn = calculate_metrics(y_test, y_pred_knn)

# Simpan trained model menggunakan pickle
model_path = "trained_knn_model.pkl"
with open(model_path, 'wb') as model_file:
    pickle.dump(knn, model_file)

st.success("Model KNN berhasil dilatih dan disimpan.")

# Visualisasi Distribusi Produk
st.subheader("Distribusi Produk")
fig, ax = plt.subplots(figsize=(8, 6))
product_counts = data['Product line'].value_counts()
bars = product_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Distribusi Produk")
ax.set_xlabel("Product Line")
ax.set_ylabel("Jumlah Penjualan")
plt.xticks(rotation=45)

# Tambahkan label angka di atas batang
for bar in ax.patches:
    ax.annotate(f'{int(bar.get_height())}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color='black')

st.pyplot(fig)

# Prediksi Penjualan Bulan Depan
next_month = df['Month'].max() + 1 if df['Month'].max() < 12 else 1
future_data = pd.DataFrame({
    'Month': [next_month] * len(df['Product line'].unique()),
    'Product line': df['Product line'].unique(),
    'Average_Unit_Price': df.groupby('Product line')['Average_Unit_Price'].mean().values
})

X_future = future_data[['Month', 'Average_Unit_Price']]
future_data_scaled = scaler.transform(X_future)

future_data['Predicted_Quantity'] = knn.predict(future_data_scaled)
future_data['Recommended_Quantity'] = np.ceil(future_data['Predicted_Quantity'] * 1.1)

st.subheader("Prediksi Penjualan dan Rekomendasi Restock")
fig, ax = plt.subplots(figsize=(10, 6))

# Tentukan lebar bar
bar_width = 0.35
index = np.arange(len(future_data['Product line']))

# Bar Predicted Quantity
bars1 = ax.bar(index, future_data['Predicted_Quantity'], bar_width, label='Predicted Quantity', color='blue')

# Bar Recommended Quantity (disandingkan)
bars2 = ax.bar(index + bar_width, future_data['Recommended_Quantity'], bar_width, label='Recommended Quantity', color='orange')

# Tambahkan label dan judul
ax.set_xlabel("Product Line")
ax.set_ylabel("Quantity")
ax.set_title("Prediksi dan Rekomendasi Restock")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(future_data['Product line'], rotation=30, ha='right')
ax.legend(loc='lower right')

# Tambahkan label nilai di atas setiap bar
for bar_group in (bars1, bars2):
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', fontsize=10, color='black')

st.pyplot(fig)