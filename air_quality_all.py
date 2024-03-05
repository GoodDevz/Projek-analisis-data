import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
sns.set(style='whitegrid')

# Konfigurasi judul halaman
st.set_page_config(page_title="Analisis Kualitas Udara 12 Stasiun di Beijing oleh gooddev")
# Muat dataset
data = pd.read_csv('air_quality_all.csv')

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara: 12 Stasiun di Beijing')

# Deskripsi
st.write('Dashboard ini menyediakan cara interaktif untuk menjelajahi data kualitas udara, khususnya fokus pada tingkat PM10 dan hubungannya dengan berbagai kondisi cuaca.')

# Tentang saya
st.markdown("""
### Tentang Saya
- **Nama**: Bagus Purnomo
- **Alamat Email**: Baguspurnomo770@gmail.com
- **ID Dicoding**: gooddev

### Gambaran Proyek
Dashboard ini menyajikan analisis data kualitas udara, terutama fokus pada tingkat PM10, dari 12 stasiun. Proyek ini bertujuan untuk mengungkap tren, variasi musiman, dan dampak kondisi cuaca yang berbeda terhadap kualitas udara. Wawasan dari analisis ini dapat berharga untuk studi lingkungan dan pemantauan kesehatan masyarakat.
""")

# Menambahkan sidebar untuk input interaktif
st.sidebar.header('Fitur Input Pengguna')

# Biarkan pengguna memilih stasiun untuk melihat data
selected_station = st.sidebar.selectbox('Pilih Stasiun', list(data['station'].unique()))

# Filter data berdasarkan stasiun yang dipilih 
data_filtered = data[(data['station'] == selected_station)] 

# Menampilkan statistik data
st.subheader('Ikhtisar Data untuk Periode Terpilih')
st.write(data_filtered.describe())

# Heatmap korelasi untuk bulan terpilih
st.subheader('Heatmap Korelasi Indikator Kualitas Udara')
corr = data_filtered[['PM10', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
plt.title('Heatmap Korelasi')
st.pyplot(fig)

# Analisis Tren Musiman
st.subheader('Analisis Tren Musiman')
tren_musiman = data.groupby('month')['PM10'].mean()
fig, ax = plt.subplots()
tren_musiman.plot(kind='bar', color='lightcoral', ax=ax)
plt.title('Rata-rata Tingkat PM10 Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata PM10')
st.pyplot(fig)

# Tingkat PM10 Harian
st.subheader('Tingkat PM10 Harian')
fig, ax = plt.subplots()
ax.plot(data_filtered['day'], data_filtered['PM10'],color='lightcoral')
plt.xlabel('Hari dalam Bulan')
plt.ylabel('Konsentrasi PM10')
st.pyplot(fig)

# Distribusi Polutan
st.subheader('Distribusi Polutan')
selected_pollutant = st.selectbox('Pilih Polutan', ['PM10', 'PM10', 'SO2', 'NO2', 'CO'])
fig, ax = plt.subplots()
sns.boxplot(x='month', y=selected_pollutant, data=data[data['station'] == selected_station],color='lightcoral', ax=ax)
st.pyplot(fig)

# Dekomposisi Seri Waktu PM10
st.subheader('Dekomposisi Seri Waktu PM10')
try:
    data_filtered['PM10'].ffill(inplace=True)
    dekomposisi = seasonal_decompose(data_filtered['PM10'], model='additive', period=24) # Sesuaikan periode sesuai kebutuhan
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    dekomposisi.trend.plot(ax=ax1, title='Tren',color='lightcoral')
    dekomposisi.seasonal.plot(ax=ax2, title='Musiman',color='lightcoral')
    dekomposisi.resid.plot(ax=ax3, title='Residu',color='lightcoral')
    plt.tight_layout()
    st.pyplot(fig)
except ValueError as e:
    st.error("Tidak dapat melakukan dekomposisi seri waktu: " + str(e))

# Heatmap Rata-rata Harian
st.subheader('Rata-rata Harian PM10')
try:
    # Pastikan tipe data yang benar dan tangani nilai yang hilang
    data['hour'] = data['hour'].astype(int)
    data['PM10'] = pd.to_numeric(data['PM10'], errors='coerce')
    data['PM10'].ffill(inplace=True)

    # Hitung rata-rata harian
    rata_rata_harian = data.groupby('hour')['PM10'].mean()

    # Plotting
    fig, ax = plt.subplots()
    sns.heatmap([rata_rata_harian.values], ax=ax, cmap='coolwarm')
    plt.title('Rata-rata Harian PM10')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Kesalahan dalam plotting rata-rata harian: {e}")

# Analisis Arah Angin
st.subheader('Analisis Arah Angin')
data_angin = data_filtered.groupby('wd')['PM10'].mean()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, len(data_angin))
bars = ax.bar(theta, data_angin.values, align='center', alpha=0.5,color='lightcoral')
plt.title('Tingkat PM10 berdasarkan Arah Angin')
st.pyplot(fig)

# Hujan vs. Kualitas Udara
st.subheader('Temperatur vs. Tingkat O3')
fig, ax = plt.subplots()
sns.scatterplot(x='TEMP', y='O3', data=data_filtered, ax=ax,color='lightcoral')
plt.title('Temperatur vs. Tingkat O3')
st.pyplot(fig)

# Heatmap Korelasi - Interaktif
st.subheader('Heatmap Korelasi Interaktif')
selected_columns = st.multiselect('Pilih Kolom untuk Korelasi', data.columns, default=['PM10', 'NO2', 'PM2.5', 'O3'])
corr = data[selected_columns].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax , cmap="coolwarm")
st.pyplot(fig)

# Kesimpulan
st.subheader('Kesimpulan')
st.write("""
- Dashboard ini menyediakan analisis yang mendalam dan interaktif tentang data kualitas udara.
- Berbagai visualisasi menawarkan wawasan tentang tingkat PM10, distribusinya, dan faktor-faktor yang memengaruhinya.
- Tren musiman dan dampak kondisi cuaca dan polutan yang berbeda terhadap kualitas udara jelas digambarkan.
- Pengguna dapat menjelajahi data secara dinamis untuk memahami lebih dalam tren kualitas udara.
""")
