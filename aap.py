from flask import Flask, request, jsonify
import numpy as np
import joblib
from hpelm import ELM
# import mysql.connector

app = Flask(__name__)

# Load model dan alat bantu
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
elm = ELM(10, encoder.transform([[0]]).shape[1], classification="c")
elm.load("elm_model.h5")

# Koneksi database MySQL
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",  # sesuaikan password
#     database="db_penyakit"
# )
# cursor = db.cursor()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = [
            float(data["umur"]),
            float(data["jenis_kelamin"]),
            float(data["nyeri_dada"]),
            float(data["tekanan_darah"]),
            float(data["kolesterol"]),
            float(data["gula_darah"]),
            float(data["ekg"]),
            float(data["detak_jantung"]),
            float(data["sakit_dada_aktivitas"]),
            float(data["atribut_hasil_prediksi_penyakit_jantung"])
        ]

        # Scale
        input_scaled = scaler.transform([input_data])

        # Predict
        result = elm.predict(input_scaled).argmax(axis=1)[0]

        label_map = {
            0: "Tidak ada penyakit jantung",
            1: "Penyakit jantung stadium 1",
            2: "Penyakit jantung stadium 2",
            3: "Penyakit jantung stadium 3",
            4: "Penyakit jantung stadium 4"
        }
        hasil_prediksi = label_map.get(result, "Tidak diketahui")

        # Simpan ke MySQL
        # sql = """
        #     INSERT INTO prediksi_penyakit (
        #         umur, jenis_kelamin, nyeri_dada, tekanan_darah, kolesterol,
        #         gula_darah, ekg, detak_jantung, angina, oldpeak, slope,
        #         thal, sakit_dada_aktivitas, hasil_prediksi
        #     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        # """
        # cursor.execute(sql, (*input_data, hasil_prediksi))
        # db.commit()

        return jsonify({
            "status": "success",
            "prediction": hasil_prediksi
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
