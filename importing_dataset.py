import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 📌 1. Veri Setini Yükleme
# ==============================================
df = pd.read_csv('Updated_Data.csv')

# ==============================================
# 📌 2. Veri Setinin İlk Görünümü
# ==============================================
print("\n🔹 İlk 5 Satır:")
print(df.head())

# ==============================================
# 📌 3. Veri Setinin Boyutu
# ==============================================
print("\n🔹 Veri Setinin Boyutu (satır, sütun):", df.shape)

# ==============================================
# 📌 4. Veri Setinin Genel Bilgileri
# ==============================================
print("\n🔹 Veri Seti Bilgisi:")
df.info()

# ==============================================
# 📌 5. İstatistiksel Özet
# ==============================================
print("\n🔹 İstatistiksel Özellikler:")
print(df.describe().T)

# ==============================================
# 📌 6. Eksik Veri Kontrolü
# ==============================================
print("\n🔹 Eksik Değer Kontrolü:")
null_counts = df.isnull().sum()
for col, count in null_counts.items():
    if count > 0:
        print(f"⚠️  Sütun '{col}' içinde {count} eksik değer var.")

# Eksik verileri temizleme
df = df.dropna()
print("\n✅ Eksik veriler temizlendi.")
print("Güncellenmiş veri setinin boyutu:", df.shape)

# ==============================================
# 📌 7. Benzersiz Değerlerin Sayısı
# ==============================================
print("\n🔹 Her Sütundaki Benzersiz Değer Sayısı:")
print(df.nunique())

# ==============================================
# 📌 8. Tarih Verisini Ayırma
# ==============================================
# Tarih sütununun parçalanması
parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

# Gereksiz sütunların kaldırılması
df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, inplace=True)

# ==============================================
# 📌 9. Sütunları Veri Tipine Göre Ayırma
# ==============================================
floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print("Kategorik Değişkenler:", objects)
print("Sayısal (Float) Değişkenler:", floats)

# ==============================================
# 📌 10. Kategorik Değişkenlerin Görselleştirilmesi (2 Satır, 2 Sütun)
# ==============================================
categorical_columns = ['Education', 'Marital_Status']
accepted_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

# Eğitim ve Medeni Durum Grafikleri
for i, col in enumerate(categorical_columns):
    sb.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(f"{col} Dağılımı")
    axes[i].tick_params(axis='x', rotation=30)

# Accepted Kampanyaları Tek Grafikte Gösterme
df_melted = df.melt(value_vars=accepted_columns, var_name="Accepted Campaigns", value_name="Accepted")
sb.countplot(x="Accepted Campaigns", hue="Accepted", data=df_melted, ax=axes[2])
axes[2].set_title("Accepted Campaigns Dağılımı")
axes[2].tick_params(axis='x', rotation=30)

# Boş kalan grafiği kaldır
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
# ==============================================
# 📌 12. Kategorik Değişkenlerin 'Response' Sütununa Göre Dağılımı (2 Satır, 2 Sütun)
# ==============================================
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))  # Boş grafik olmaması için 1 satır, 2 sütun
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    sb.countplot(x=col, hue='Response', data=df, ax=axes[i])
    axes[i].set_title(f"{col} - Response Dağılımı")
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# ==============================================
# 📌 13. Kategorik Değişkenleri Sayısala Çevirme (Label Encoding)
# ==============================================
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ==============================================
# 📌 14. Korelasyon Haritası (Heatmap)
# ==============================================
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.title("Korelasyon Matrisi")
plt.show()

# ==============================================
# 📌 15. Veriyi Ölçeklendirme (StandardScaler)
# ==============================================
scaler = StandardScaler()
data = scaler.fit_transform(df)