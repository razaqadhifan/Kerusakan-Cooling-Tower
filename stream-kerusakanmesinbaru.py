import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import date

# Load model dan scaler
model_kerusakan = pickle.load(open('model_kerusakan_ann_terbaru.sav', 'rb'))
scaler = pickle.load(open('scaler_ann_terbaru.sav', 'rb'))

st.set_page_config(page_title="Prediksi Kerusakan Mesin Cooling Tower", layout="wide")

# Logo atau gambar dan judul
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("Logo-UBJ.png", width=500)
with col_title:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='margin-bottom: 0;'>❄️ Prediksi Kerusakan Mesin Cooling Tower</h1>
        <p style='margin-top: 0; font-size: 18px;'>Fakultas Teknik Industri - Universitas Bhayangkara Jakarta Raya</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar untuk navigasi
with st.sidebar:
    st.header("📋 Menu")
    page = st.radio("Navigasi", ["Prediksi", "Riwayat"])

if page == "Prediksi":
    st.subheader("🛠️ Form Input Data Mesin")

    st.markdown("""
    <div style='background-color:  #e7f3fe; padding: 10px; border-radius: 8px; border-left: 5px solid #e7f3fe;'>
        📌 <i>Apabila muncul keabnormalan pada mesin Cooling Tower, mohon hubungi leader terkait.</i>
    </div>
    """, unsafe_allow_html=True)

    operator = st.text_input("Nama Operator")
    tanggal = st.date_input("Tanggal Input", value=date.today())

    if 'Frekuensi_Fan_1' not in st.session_state:
        st.session_state['Frekuensi_Fan_1'] = 40.0
    if 'Frekuensi_Fan_2' not in st.session_state:
        st.session_state['Frekuensi_Fan_2'] = 40.0
    if 'Pressure_Gauge' not in st.session_state:
        st.session_state['Pressure_Gauge'] = 1.5
    if 'Inlet' not in st.session_state:
        st.session_state['Inlet'] = 30.0
    if 'Outlet' not in st.session_state:
        st.session_state['Outlet'] = 30.0

    col1, col2, col3 = st.columns(3)
    with col1:
        Fan_and_Motor = st.selectbox('Fan & Motor', ["1 - Tidak Noise", "2 - Noise"], help="Pilih 2 jika terdengar suara bising pada kipas.")
        Frekuensi_Fan_1 = st.number_input('Frekuensi Fan 1', min_value=0.0, step=0.1, value=st.session_state['Frekuensi_Fan_1'], help="Frekuensi ideal biasanya sekitar 40 Hz.")
        Frekuensi_Fan_2 = st.number_input('Frekuensi Fan 2', min_value=0.0, step=0.1, value=st.session_state['Frekuensi_Fan_2'], help="Frekuensi ideal biasanya sekitar 40 Hz.")
    with col2:
        Pressure_Gauge = st.number_input('Pressure Gauge (kgf/cm²)', min_value=0.0, step=0.1, value=st.session_state['Pressure_Gauge'], help="Tekanan normal berkisar antara 1.0 hingga 2.5 kgf/cm².")
        Pipe_and_Strainer = st.selectbox('Pipe & Strainer', ["1 - Tidak Bermasalah", "2 - Bermasalah"], help="Pilih 2 jika ada penyumbatan atau kerusakan pada pipa atau saringan.")
        Inlet = st.number_input('Inlet (°C)', min_value=0.0, step=0.1, value=st.session_state['Inlet'], help="Suhu normal sekitar 28 - 32 °C.")
    with col3:
        Outlet = st.number_input('Outlet (°C)', min_value=0.0, step=0.1, value=st.session_state['Outlet'], help="Suhu normal sekitar 28 - 32 °C.")

    if st.button('🎲 Simulasi Data Acak'):
        st.session_state['Frekuensi_Fan_1'] = float(np.random.uniform(35, 45))
        st.session_state['Frekuensi_Fan_2'] = float(np.random.uniform(35, 45))
        st.session_state['Pressure_Gauge'] = float(np.random.uniform(0.5, 3.0))
        st.session_state['Inlet'] = float(np.random.uniform(26, 34))
        st.session_state['Outlet'] = float(np.random.uniform(26, 34))

    hasil_prediksi = ''

    if st.button('🔍 Prediksi Kerusakan'):
        try:
            fan_motor_val = 2 if "2" in Fan_and_Motor else 1
            pipe_strainer_val = 2 if "2" in Pipe_and_Strainer else 1

            input_data = [
                fan_motor_val,
                Frekuensi_Fan_1,
                Frekuensi_Fan_2,
                Pressure_Gauge,
                pipe_strainer_val,
                Inlet,
                Outlet
            ]

            arr_data = np.array([input_data])
            scaled_data = scaler.transform(arr_data)

            hasil = model_kerusakan.predict(scaled_data)[0]
            proba = model_kerusakan.predict_proba(scaled_data)[0][1]

            st.caption(f"Probabilitas kerusakan: {proba:.2f} (Threshold default: 0.50)")
            st.markdown("<sub>Probabilitas menunjukkan seberapa yakin model bahwa mesin mengalami kerusakan. Semakin mendekati 1.00, semakin besar kemungkinan mesin bermasalah.</sub>", unsafe_allow_html=True)

            if hasil == 1:
                hasil_prediksi = '⚠️ Mesin Perlu Perhatian'
                st.error(hasil_prediksi)

                catatan = []
                if fan_motor_val == 2:
                    catatan.append("- Fan & Motor mengalami noise, periksa bearing dan getaran.")
                if pipe_strainer_val == 2:
                    catatan.append("- Pipa dan strainer bermasalah, cek aliran dan kemungkinan penyumbatan.")
                if Pressure_Gauge < 1.0 or Pressure_Gauge > 2.5:
                    catatan.append("- Pressure Gauge berada di luar batas normal (1.0 - 2.5 kgf/cm²), periksa tekanan sistem.")
                if Inlet < 28.0:
                    catatan.append("- Suhu Inlet terlalu rendah, pastikan tidak ada masalah pada sistem pendingin.")
                if Inlet > 32.0:
                    catatan.append("- Suhu Inlet melebihi batas normal, cek sistem pendingin atau beban mesin.")
                if Outlet < 28.0:
                    catatan.append("- Suhu Outlet terlalu rendah, pastikan tidak ada masalah pada sistem pendingin.")
                if Outlet > 32.0:
                    catatan.append("- Suhu Outlet terlalu tinggi, cek efisiensi pendinginan dan aliran keluar.")

                if catatan:
                    st.markdown("**Hal-hal yang perlu diperhatikan:**")
                    for item in catatan:
                        st.markdown(item)
            else:
                hasil_prediksi = '✅ Mesin Aman'
                st.success(hasil_prediksi)

            row = {
                "Tanggal": [tanggal],
                "Operator": [operator],
                "Fan & Motor": [Fan_and_Motor],
                "Frekuensi Fan 1": [Frekuensi_Fan_1],
                "Frekuensi Fan 2": [Frekuensi_Fan_2],
                "Pressure Gauge": [Pressure_Gauge],
                "Pipe & Strainer": [Pipe_and_Strainer],
                "Inlet": [Inlet],
                "Outlet": [Outlet],
                "Hasil Prediksi": ["Perlu Perhatian" if hasil == 1 else "Aman"],
                "Probabilitas": [round(proba, 2)]
            }

            df_new = pd.DataFrame(row)

            if os.path.exists("riwayat_prediksi.xlsx"):
                df_old = pd.read_excel("riwayat_prediksi.xlsx")
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new

            df_all.to_excel("riwayat_prediksi.xlsx", index=False)
            st.success("📁 Data berhasil disimpan ke riwayat!")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

elif page == "Riwayat":
    st.subheader("📈 Riwayat Prediksi Mesin")
    if os.path.exists("riwayat_prediksi.xlsx"):
        df = pd.read_excel("riwayat_prediksi.xlsx")

        if st.checkbox("Saya yakin ingin menghapus semua riwayat"):
            if st.button("🗑️ Hapus Semua Riwayat Sekarang"):
                os.remove("riwayat_prediksi.xlsx")
                st.warning("Seluruh riwayat berhasil dihapus.")
                st.stop()

        st.markdown("### Filter Riwayat Berdasarkan Tanggal")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Dari Tanggal", value=date.today())
        with col2:
            end_date = st.date_input("Sampai Tanggal", value=date.today())

        df["Tanggal"] = pd.to_datetime(df["Tanggal"]).dt.date
        df_filtered = df[(df["Tanggal"] >= start_date) & (df["Tanggal"] <= end_date)]

        st.markdown("### 📊 Grafik Probabilitas Kerusakan")
        chart_data = df_filtered[['Tanggal', 'Probabilitas']].copy()
        chart_data['Tanggal'] = pd.to_datetime(chart_data['Tanggal'])
        chart_data.sort_values('Tanggal', inplace=True)
        st.line_chart(chart_data.set_index('Tanggal'))

        st.markdown("### 🔢 Ringkasan Statistik")
        st.write(f"Jumlah prediksi: {len(df_filtered)}")
        st.write(f"Mesin Aman: {(df_filtered['Hasil Prediksi'] == 'Aman').sum()}")
        st.write(f"Mesin Perlu Perhatian: {(df_filtered['Hasil Prediksi'] == 'Perlu Perhatian').sum()}")

        st.dataframe(df_filtered, use_container_width=True)
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")
        st.warning("Silakan lakukan prediksi terlebih dahulu untuk melihat riwayat.")
