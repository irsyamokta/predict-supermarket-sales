import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

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
    accuracy = (1 - rmse/np.mean(y_true)) * 100
    return rmse, accuracy

st.title("Prediksi Penjualan Supermarket dengan KNN & SVM")
st.sidebar.header("Upload Dataset Anda")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.warning("Silakan unggah file CSV untuk melanjutkan.")
    st.stop()

st.write("Dataset Awal", data.head())
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

svm = SVR(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

rmse_knn, accuracy_knn = calculate_metrics(y_test, y_pred_knn)
rmse_svm, accuracy_svm = calculate_metrics(y_test, y_pred_svm)

model_option = st.selectbox("Pilih Model Prediksi", ["KNN", "SVM"])

if model_option == "KNN":
    rmse = rmse_knn
    accuracy_value = accuracy_knn
    model = knn
else:
    rmse = rmse_svm
    accuracy_value = accuracy_svm
    model = svm

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

next_month = df['Month'].max() + 1 if df['Month'].max() < 12 else 1
future_data = pd.DataFrame({
    'Month': [next_month] * len(df['Product line'].unique()),
    'Product line': df['Product line'].unique(),
    'Average_Unit_Price': df.groupby('Product line')['Average_Unit_Price'].mean().values
})

X_future = future_data[['Month', 'Average_Unit_Price']]
future_data_scaled = scaler.transform(X_future)

future_data['Predicted_Quantity'] = model.predict(future_data_scaled)
future_data['Recommended_Quantity'] = np.ceil(future_data['Predicted_Quantity'] * 1.1)

st.subheader("Prediksi Penjualan Bulan Depan")
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(future_data['Product line'], future_data['Predicted_Quantity'], color='blue')
ax1.set_xlabel("Product Line")
ax1.set_ylabel("Prediksi Quantity")
ax1.tick_params(axis='x', rotation=45)
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, color='black')
st.pyplot(fig1)

st.subheader("Rekomendasi Restock")
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars2 = ax2.bar(future_data['Product line'], future_data['Recommended_Quantity'], color='orange')
ax2.set_xlabel("Product Line")
ax2.set_ylabel("Recommended Quantity")
ax2.tick_params(axis='x', rotation=45)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, color='black')
st.pyplot(fig2)
