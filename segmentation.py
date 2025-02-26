# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.manifold import TSNE  # t-SNE algoritması için
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Veri ön işleme için
from sklearn.cluster import KMeans  # Kümeleme algoritması için
import warnings

# Uyarıları kapatıyoruz, gereksiz mesajları görmemek için
warnings.filterwarnings('ignore')

# 1️⃣ VERİYİ YÜKLEME
# Veri setini CSV dosyasından okuyoruz
df = pd.read_csv('Updated_Data.csv')

# 2️⃣ KATEGORİK VERİLERİ SAYISALA ÇEVİRME
# "Education" ve "Marital_Status" sütunları kategorik olduğu için sayısal hale getiriyoruz.
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])  # Örneğin "Graduation" → 0, "Master" → 1, vs.
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])  # Örneğin "Single" → 0, "Married" → 1, vs.

# 3️⃣ TARİH VERİSİNİ SAYISALLAŞTIRMA
# "Dt_Customer" sütunu müşteri kayıt tarihini içeriyor, bunu gün sayısına çeviriyoruz.
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)  # Gün-Ay-Yıl formatında dönüştürüyoruz
df['Customer_Age'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days  # En yeni tarihten itibaren kaç gün geçtiğini hesaplıyoruz
df.drop(columns=['Dt_Customer'], inplace=True)  # Orijinal tarih sütununu siliyoruz, artık ihtiyacımız yok

# 4️⃣ EKSİK VERİLERİ DOLDURMA
# Eğer veri setinde eksik (NaN) değerler varsa bunları ortalama (mean) ile dolduruyoruz.
df.fillna(df.mean(), inplace=True)

# 5️⃣ VERİYİ ÖLÇEKLENDİRME
# Makine öğrenmesi algoritmalarının daha iyi çalışması için veriyi standart ölçeklendirme yapıyoruz.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Standart ölçeklendirme, tüm değişkenleri aynı ölçeğe getirir

# 6️⃣ t-SNE MODELİ OLUŞTURMA VE GÖRSELLEŞTİRME
# Veriyi iki boyutlu hale getiriyoruz (t-SNE kullanarak)
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(df_scaled)  # t-SNE dönüşümünü gerçekleştiriyoruz

# t-SNE sonucunu scatter plot (dağılım grafiği) ile gösteriyoruz
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.show()

# 7️⃣ OPTİMAL KÜME SAYISINI BELİRLEME (K-Means ile)
# K-Means algoritmasının en uygun küme sayısını belirlemek için "Dirsek Yöntemi"ni kullanıyoruz.

error = []  # Her küme sayısı için hata değerlerini tutacağımız liste
for n_clusters in range(1, 21):  # 1'den 20'ye kadar farklı küme sayıları deniyoruz
    model = KMeans(init='k-means++',
                   n_clusters=n_clusters,  # Küme sayısı (k)
                   max_iter=500,  # Maksimum iterasyon sayısı
                   random_state=22)  # Rastgelelik için sabit seed değeri
    model.fit(df_scaled)  # Modeli eğitiyoruz
    error.append(model.inertia_)  # Kümeleme hata değerini listeye ekliyoruz

# 8️⃣ KÜME SAYISI ANALİZİ GRAFİĞİ
# Küme sayısına karşılık hata değerlerini çiziyoruz (Dirsek Yöntemi)
plt.figure(figsize=(10, 5))
sb.lineplot(x=range(1, 21), y=error)  # Hata değerlerini çiziyoruz
sb.scatterplot(x=range(1, 21), y=error)  # Verileri nokta olarak da gösteriyoruz
plt.show()

# 9️⃣ OPTİMAL K = 5 KÜMELEME MODELİ OLUŞTURMA
# Grafiğe göre en uygun küme sayısını belirliyoruz (Örneğin k=5 en uygun noktaysa)
model = KMeans(init='k-means++',
               n_clusters=5,  # En uygun küme sayısı
               max_iter=500,  # Maksimum iterasyon sayısı
               random_state=22)  # Rastgelelik için sabit seed değeri
segments = model.fit_predict(df_scaled)  # Modeli çalıştırıyoruz ve hangi noktanın hangi kümeye ait olduğunu alıyoruz

# 1️⃣0️⃣ KÜMELEME SONUÇLARINI GÖRSELLEŞTİRME
plt.figure(figsize=(7, 7))
# t-SNE ile elde edilen veriyi DataFrame'e çeviriyoruz
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})

# Kümeleme sonuçlarını scatter plot olarak çiziyoruz
sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne)
plt.show()
