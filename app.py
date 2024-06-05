from flask import Flask, render_template, Markup
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def index():
    # Membaca data dari file CSV
    data = pd.read_csv('data_penjualan.csv')

    # Konversi kolom tanggal ke tipe datetime
    data['tanggal'] = pd.to_datetime(data['tanggal'])

    # Pembersihan Data
    data = data.dropna()

    # Transformasi Data
    data = pd.get_dummies(data, columns=['kategori'])

    # Exploratory Data Analysis (EDA)
    fig = px.scatter(data, x='jumlah', y='harga', title='Hubungan Antara Jumlah Barang Terjual dan Harga', trendline="ols")

    # Mengonversi plot ke format HTML
    graph_html = pio.to_html(fig, full_html=False)

    # Mengonversi tabel data ke format HTML
    table_html = data.to_html(classes='table table-striped', index=False)

    return render_template('index.html', graph_html=Markup(graph_html), table_html=Markup(table_html))

if __name__ == '__main__':
    app.run(debug=True)
