from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import datetime
import os

app = Flask(__name__)

def get_komoditas_list():
    data = pd.read_csv("pangann.csv")
    data['Komoditas'] = data['Komoditas'].str.strip()
    return sorted(data['Komoditas'].unique())

def load_data():
    data = pd.read_csv("pangann.csv")

    data['Komoditas'] = data['Komoditas'].str.strip()
    data['Harga_Clean'] = pd.to_numeric(
        data['Harga'].astype(str).str.replace('Rp', '').str.replace(',', ''),
        errors='coerce'
    )

    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    data['Bulan_Angka'] = data['Bulan'].map(bulan_map)
    data['Tanggal'] = pd.to_datetime(dict(year=data['Tahun'], month=data['Bulan_Angka'], day=1))
    data['X_Time'] = data['Tanggal'].apply(lambda d: d.toordinal())

    return data

def train_model(data, komoditas):
    df = data[data['Komoditas'] == komoditas].copy()

    x = df['X_Time'].values
    y = df['Harga_Clean'].values

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    return slope, intercept, df


def predict_price(date, slope, intercept):
    x_pred = date.toordinal()
    return slope * x_pred + intercept


def plot_prediction(df, slope, intercept, predict_dates):
    x = df['X_Time']
    y = df['Harga_Clean']

    plt.figure(figsize=(10,5))
    plt.scatter(x, y, label="Data Asli")

    line_x = np.linspace(min(x), max(x) + 2000, 200)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, label="Regresi Linear")

    for d in predict_dates:
        xp = d.toordinal()
        yp = predict_price(d, slope, intercept)
        plt.scatter(xp, yp, marker='x', s=100, label=f"Prediksi {d.strftime('%b %Y')}")

    plt.xlabel("Ordinal Time")
    plt.ylabel("Harga")
    plt.legend()
    plt.tight_layout()

    if not os.path.exists("static"):
        os.mkdir("static")

    plt.savefig("static/plot.png")
    plt.close()


@app.route('/')
def index():
    komoditas_list = get_komoditas_list()
    return render_template("index.html", komoditas_list=komoditas_list)


@app.route('/predict', methods=['POST'])
def predict():
    komoditas = request.form['komoditas']
    tahun = int(request.form['tahun'])
    bulan = int(request.form['bulan'])

    data = load_data()
    slope, intercept, df = train_model(data, komoditas)

    target_date = datetime.date(tahun, bulan, 1)
    hasil_pred = predict_price(target_date, slope, intercept)

    plot_prediction(df, slope, intercept, [target_date])

    komoditas_list = get_komoditas_list()
 
    return render_template(
        "index.html",
        komoditas_list=komoditas_list,
        hasil_prediksi=f"{hasil_pred:,.0f}",
        komoditas=komoditas,
        bulan=bulan,
        tahun=tahun
    )



# ---------------------------- RUN APP -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
