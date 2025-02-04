import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load dataset
def load_data():
    data = pd.read_csv("dataset/supermarket_sales - Sheet1.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data

# Preprocessing
def preprocess_data(data):
    # Tangani missing value
    data = data.copy()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].mean(), inplace=True)
    
    # Hapus duplikat
    data.drop_duplicates(inplace=True)
    
    # Grouping seperti di notebook
    grouped_data = data.groupby(['Product line', 'Month']).agg(
        Total_Quantity=('Quantity', 'sum'),
        Average_Unit_Price=('Unit price', 'mean')
    ).reset_index()
    
    return grouped_data

# Function untuk menghitung metrics yang konsisten dengan notebook
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = (1 - rmse/np.mean(y_true)) * 100
    return rmse, accuracy

# Load data
data = load_data()

# Preprocessing
df = preprocess_data(data)

# Prepare features
X = df[['Month', 'Average_Unit_Price']]
y = df['Total_Quantity']

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models dengan parameter yang sama seperti notebook
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

svm = SVR(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Calculate metrics
rmse_knn, accuracy_knn = calculate_metrics(y_test, y_pred_knn)
rmse_svm, accuracy_svm = calculate_metrics(y_test, y_pred_svm)

# Streamlit UI
st.title("Prediksi Penjualan Supermarket dengan KNN & SVM")

# Menampilkan dataset awal
st.write("Dataset Awal", data.head())

# Pilih model
model_option = st.selectbox("Pilih Model Prediksi", ["KNN", "SVM"])

# Prediksi dan evaluasi
if model_option == "KNN":
    rmse = rmse_knn
    accuracy_value = accuracy_knn
else:
    rmse = rmse_svm
    accuracy_value = accuracy_svm

# Menampilkan hasil RMSE dan Akurasi Model
st.subheader("Evaluasi Model")
st.write(f"**Model yang dipilih: {model_option}**")
st.write(f"📉 **RMSE: {rmse:.4f}**")
st.write(f"✅ **Akurasi: {accuracy_value:.2f}%**")

# Visualisasi Distribusi Produk
st.subheader("Distribusi Produk")
fig, ax = plt.subplots(figsize=(8, 6))
data['Product line'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Distribusi Produk")
ax.set_xlabel("Product Line")
ax.set_ylabel("Jumlah Penjualan")
plt.xticks(rotation=45)
st.pyplot(fig)

# Prediksi untuk bulan berikutnya
next_month = df['Month'].max() + 1 if df['Month'].max() < 12 else 1
future_data = pd.DataFrame({
    'Month': [next_month] * len(df['Product line'].unique()),
    'Product line': df['Product line'].unique()
})

# Membuat label encoder yang konsisten
product_line_encoder = {label: idx for idx, label in enumerate(df['Product line'].unique())}

# Prediksi untuk bulan berikutnya
next_month = df['Month'].max() + 1 if df['Month'].max() < 12 else 1
future_data = pd.DataFrame({
    'Month': [next_month] * len(df['Product line'].unique()),
    'Product line': df['Product line'].unique(),
    'Average_Unit_Price': df.groupby('Product line')['Average_Unit_Price'].mean().values
})

# Persiapkan fitur untuk prediksi
X_future = future_data[['Month', 'Average_Unit_Price']]
future_data_scaled = scaler.transform(X_future)

# Prediksi untuk bulan berikutnya
if model_option == "KNN":
    future_data['Predicted_Quantity'] = knn.predict(future_data_scaled)
else:
    future_data['Predicted_Quantity'] = svm.predict(future_data_scaled)

future_data['Recommended_Quantity'] = np.ceil(future_data['Predicted_Quantity'] * 1.1)

# Visualisasi Prediksi dan Rekomendasi
st.subheader("Prediksi Penjualan Bulan Depan")
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(future_data['Product line'], future_data['Predicted_Quantity'], color='blue')
ax1.set_xlabel("Product Line")
ax1.set_ylabel("Prediksi Quantity")
ax1.tick_params(axis='x', rotation=45)

# Tambahkan label angka di atas batang
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, color='black')

st.pyplot(fig1)

# Rekomendasi Restock
st.subheader("Rekomendasi Restock")
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars2 = ax2.bar(future_data['Product line'], future_data['Recommended_Quantity'], color='orange')
ax2.set_xlabel("Product Line")
ax2.set_ylabel("Recommended Quantity")
ax2.tick_params(axis='x', rotation=45)

# Tambahkan label angka di atas batang
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, color='black')

st.pyplot(fig2)