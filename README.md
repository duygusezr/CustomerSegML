# Veri Kümeleme ve Segmentasyon Projesi

## 📌 Proje Açıklaması
Bu proje, bir veri kümesi üzerinde ön işleme, görselleştirme ve segmentasyon işlemlerini gerçekleştirmektedir.
Proje kapsamında **Python** programlama dili ve **pandas, seaborn, scikit-learn** gibi popüler veri bilimi kütüphaneleri kullanılmıştır.

## 📂 Proje İçeriği
- **`importing_dataset.py`**: Veri setini yükleme, temizleme ve görselleştirme işlemlerini gerçekleştirir.
- **`segmentation.py`**: Ön işlenmiş veriyi kullanarak kümeleme (clustering) işlemi yapar.
- **`Updated_Data.csv`**: İşlenen veri kümesi.

---

## 🚀 Kurulum ve Kullanım
Bu projeyi çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

### 1️⃣ Gerekli Kütüphaneleri Yükleyin
Proje, aşağıdaki kütüphanelere ihtiyaç duymaktadır. Gerekli kütüphaneleri yüklemek için terminal veya komut satırında şu komutu çalıştırabilirsiniz:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2️⃣ Veri Setini Hazırlayın
`Updated_Data.csv` adlı dosyanın proje dizininde bulunduğundan emin olun.

### 3️⃣ Veri Setini İnceleme
`importing_dataset.py` dosyasını çalıştırarak veri setini inceleyebilirsiniz:
```sh
python importing_dataset.py
```
Bu betik aşağıdaki işlemleri yapar:
- Veriyi yükler.
- Eksik değerleri temizler.
- Temel istatistiksel analizleri yapar.
- Kategorik verileri sayısal hale getirir.
- Korelasyon analizi ve görselleştirme yapar.

### 4️⃣ Segmentasyon İşlemi
Veri segmentasyonu için `segmentation.py` dosyasını çalıştırın:
```sh
python segmentation.py
```
Bu betik aşağıdaki işlemleri yapar:
- t-SNE ile veri boyutunu azaltır ve görselleştirir.
- K-Means kümeleme yöntemi ile optimal küme sayısını belirler.
- Veri noktalarını kümeler halinde gösterir.

---

## 📊 Kullanılan Teknikler
### 🔹 Veri Ön İşleme
- Eksik değerleri temizleme
- Kategorik verileri sayısal hale çevirme (`LabelEncoder`)
- Veriyi ölçeklendirme (`StandardScaler`)

### 🔹 Görselleştirme
- Dağılım grafikleri (`seaborn.countplot`)
- Korelasyon matrisi (`seaborn.heatmap`)
- t-SNE görselleştirmesi (`matplotlib.scatter`)

### 🔹 Kümeleme
- **t-SNE**: Yüksek boyutlu veriyi 2D hale getirerek kümeleme için uygun hale getirme
- **K-Means**: Kümeleme modeli oluşturma ve optimal küme sayısını belirleme (Dirsek Yöntemi)

---


