import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Membaca data dari file CSV
data = pd.read_csv('data_penjualan.csv')

# Konversi kolom tanggal ke tipe datetime
data['tanggal'] = pd.to_datetime(data['tanggal'])

# 2. Pembersihan Data
# Memeriksa nilai-nilai yang hilang
print(data.isnull().sum())

# Menghapus baris yang memiliki nilai-nilai yang hilang
data = data.dropna()

# 3. Transformasi Data
data = pd.get_dummies(data, columns=['kategori'])

# Tampilkan data di terminal
print("Data Penjualan:")
print(data)

# 4. Exploratory Data Analysis (EDA)
plt.figure(figsize=(20, 10))

# Scatter plot jumlah vs harga
plt.subplot(2, 3, 1)
plt.scatter(data['jumlah'], data['harga'], label='Data')
plt.xlabel('Jumlah Barang Terjual')
plt.ylabel('Harga')
plt.title('Scatter Plot Jumlah vs Harga')
plt.legend()

# Pembagian Data
X = data[['jumlah']]
y = data['harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pemodelan Data
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Garis regresi
x_range = range(int(X['jumlah'].min()), int(X['jumlah'].max()) + 1)
y_range = model.predict(pd.DataFrame(x_range))

# Scatter plot dengan garis regresi
plt.subplot(2, 3, 2)
plt.scatter(X_test, y_test, label='Data Uji')
plt.plot(x_range, y_range, color='red', label='Garis Regresi')
plt.xlabel('Jumlah Barang Terjual')
plt.ylabel('Harga')
plt.title('Scatter Plot dengan Garis Regresi')
plt.legend()

# Histogram dari nilai aktual dan prediksi
plt.subplot(2, 3, 3)
plt.hist(y_test, alpha=0.5, label='Aktual')
plt.hist(y_pred, alpha=0.5, label='Prediksi')
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
plt.title('Histogram Harga Aktual vs Prediksi')
plt.legend()

# Scatter plot nilai aktual vs prediksi
plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred, label='Prediksi vs Aktual')
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.title('Scatter Plot Harga Aktual vs Prediksi')
plt.legend()

# Bar plot jumlah penjualan per kategori
plt.subplot(2, 3, 5)
data.groupby('kategori_A')['jumlah'].sum().plot(kind='bar', color=['blue', 'orange', 'green'])
plt.xlabel('Kategori')
plt.ylabel('Jumlah Barang Terjual')
plt.title('Bar Plot Jumlah Barang Terjual per Kategori')

# Bar plot jumlah penjualan per bulan
data['bulan'] = data['tanggal'].dt.to_period('M')
monthly_sales = data.groupby('bulan')['jumlah'].sum()

plt.subplot(2, 3, 6)
monthly_sales.plot(kind='bar', color='purple')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Barang Terjual')
plt.title('Bar Plot Jumlah Barang Terjual per Bulan')

plt.tight_layout()
plt.show()

# 5. Validasi dan Penilaian Model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Tampilkan prediksi vs nilai sebenarnya
print("Prediksi vs Nilai Sebenarnya:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)
