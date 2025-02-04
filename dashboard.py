import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load dataset
def load_data():
    data = pd.read_csv("dataset/supermarket_sales - Sheet1.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    return data

data = load_data()

# Preprocessing
def preprocess_data(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    le = LabelEncoder()
    data['Product_Label'] = le.fit_transform(data['Product line'])
    grouped_data = data.groupby(['Month', 'Product_Label'])['Quantity'].sum().reset_index()
    return grouped_data, le

df, label_encoder = preprocess_data(data)

# Train-test split
X = df[['Month', 'Product_Label']]
y = df['Quantity']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))

# Model SVM
svm = SVR()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))

# Streamlit UI
st.title("Prediksi Penjualan Supermarket dengan KNN & SVM")

# Menampilkan dataset awal
st.write("Dataset Awal", data.head())

# Pilih model
model_option = st.selectbox("Pilih Model Prediksi", ["KNN", "SVM"])

# Prediksi
y_pred = y_pred_knn if model_option == "KNN" else y_pred_svm
rmse = rmse_knn if model_option == "KNN" else rmse_svm

def accuracy(y_test, y_pred):
    return 1 - np.sqrt(np.mean((y_test - y_pred) ** 2)) / np.mean(y_test)

model_accuracy = accuracy(y_test, y_pred) * 100

# Diagram Distribusi Label
st.subheader("Distribusi Produk")
fig, ax = plt.subplots(figsize=(8, 6))
data['Product line'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Distribusi Label Produk")
ax.set_xlabel("Product Line")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
st.pyplot(fig)

# Menambahkan bulan berikutnya untuk prediksi
next_month = df['Month'].max() + 1 if df['Month'].max() < 12 else 1
future_data = pd.DataFrame({
    'Month': [next_month] * df['Product_Label'].nunique(),
    'Product_Label': df['Product_Label'].unique()
})

# Normalisasi data sebelum prediksi
future_data_scaled = scaler.transform(future_data)
future_data['Predicted_Quantity'] = knn.predict(future_data_scaled) if model_option == "KNN" else svm.predict(future_data_scaled)
future_data['Recommended_Quantity'] = np.ceil(future_data['Predicted_Quantity'] * 1.1)

# Konversi label kembali ke nama produk
future_data['Product line'] = label_encoder.inverse_transform(future_data['Product_Label'])

# Grafik Prediksi Penjualan (Atas)
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

# Grafik Rekomendasi Restock (Bawah)
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


