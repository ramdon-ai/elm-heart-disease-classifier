from flask import Flask, render_template, request
import numpy as np
import joblib
from hpelm import ELM
import pandas as pd

app = Flask(__name__)

# Load model dan tools
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
elm = ELM(10, encoder.transform([[0]]).shape[1], classification="c")
elm.load("elm_model.h5")

# 2. Daftar nama fitur (pastikan sesuai saat training)
feature_names = ['Umur', 'Jenis Kelamin', 'Nyeri Dada', 'Tekanan Darah', 'Kolesterol',
                 'Gula Darah', 'Hasil EKG', 'Detak Jantung', 'Sakit Dada Selama Beraktivitas', 
                 'Atribut Hasil Prediksi Penyakit Jantung']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            print("\n--- Mulai Proses Prediksi ---")
            # Ambil input dari form HTML
            input_data = [
                float(request.form["umur"]),
                float(request.form["jenis_kelamin"]),
                float(request.form["nyeri_dada"]),
                float(request.form["tekanan_darah"]),
                float(request.form["kolesterol"]),
                float(request.form["gula_darah"]),
                float(request.form["ekg"]),
                float(request.form["detak_jantung"]),
                float(request.form["sakit_dada_aktivitas"]),
                float(request.form["atribut_hasil_prediksi_penyakit_jantung"])
            ]
            print("Input mentah:", input_data)

            # Ubah ke DataFrame untuk standardisasi
            df = pd.DataFrame([input_data], columns=feature_names)
            input_scaled = scaler.transform(df)
            print("Setelah transformasi (scaled):", input_scaled)

            # Prediksi
            result = elm.predict(input_scaled).argmax(axis=1)[0]
            print("Hasil prediksi kelas (angka):", result)

            # Mapping label
            label_map = {
                0: "Tidak ada penyakit jantung",
                1: "Penyakit jantung stadium 1",
                2: "Penyakit jantung stadium 2",
                3: "Penyakit jantung stadium 3",
                4: "Penyakit jantung stadium 4"
            }
            prediction = label_map.get(result, "Tidak diketahui")
            print("Hasil prediksi akhir:", prediction)
            print("--- Proses Selesai ---\n")

        except Exception as e:
            prediction = f"Terjadi error: {e}"
            print("Terjadi error saat prediksi:", e)

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
