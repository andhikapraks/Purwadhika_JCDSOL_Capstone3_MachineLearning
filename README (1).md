# E-Commerce Customer Churn by Machine Learning (Classification Model)
Capstone Project 3: Machine Learning == E-Commerce Customer Churn <br>
Andhika Prakoso - JCDSOL - A

# **Context and Objective**

-Context-
    - Commerce X ingin mencari tahu siapa pelanggan yang akan churn, sehingga dapat melakukan antisipasi sebelum itu terjadi secara efektif dan efisien baik melalui pemberian promo ataupun cara lainnya. Sebagai seorang data scientist, kamu diminta untuk membuat permodelan yang memprediksi siapa pelanggan-pelanggan yang berpotensi churn tersebut beserta karakteristik/pertanda indikasinya.


-Objective-
    - Membuat model klasifikasi untuk memprediksi mana pelanggan yang akan Churn dari platform-nya sebelum hal tersebut terjadi
    - Mengetahui faktor/indikasi yang menjadi pertanda bagi pelanggan yang rentan akan menjadi Churn

- Key metrics -
    - F1 Score 
        - Type 1 error : False Positive --> Konsekuensi: sia-sianya promo yang diberikan, usaha dan biaya terbuang sia-sia 
        - Type 2 error : False Negative  --> Konsekuensi: kehilangan pelanggan


# **Machine Learning Development**
- Data cleaning
    - Data duplicate drop
    - Data value missing: KNN imputer
    - Data outliers: Retain in the dataset

- Data preprocessing
    - Encoding = One-hot encoding
    - Scaling = Robust Scaler
    - Data split

- Model selected
    - Dari hasil benchmark yang dilakukan ditemukan model LightGBM merupakan model dengan performa skor F1 terbaik diantara model lainnya.
    - Final score:
        - Data train: 0.764
        - Data test: 0.792
    - Parameter:
        - Class_weight: Balanced
        - Feature selection: 10 (RFE)
        - Hyperparamter tuning: - `max_bin`: 275, `num_leaves`: 21, `min_data_in_leaf`: 10, `num_iterations`: 200, `learning_rate`: 0.125

# **Impact Model to Business**

**Model mampu memprediksi klasifikasi customer yang akan Churn/Tidak churn dengan cukup akurat** 

Dari 654 pelanggan yang dijadikan data test pada evaluasi model, model LightGBM yang dibangun mampu mengklasifikasikan sebanyak; 
- 523 pelanggan yang tidak Churn (True Negative)
- 86 pelanggan Churn (True Negative)
- Meski masih ada sekitar 45 orang terklasifikasi secara kurang tepat (False Positive & False Negative) 

Artinya model mampu mendeteksi sebesar 93% pelanggan secara tepat apakah mereka termasuk pelanggan yang akan Churn & Tidak Churn.

**Model membantu e-commerce dalam penentuan pemberian promo dan penghematan biaya sekaligus**.

Mari kita ilustrasikan menggunakan Asumsi pemberian promo senilai 100 per pelanggan.

1. Pemberian promo tanpa adanya model machine learning mengharuskan e-commerce perlu memberikan promo ke seluruh pelanggan.
- Biaya yang dibutuhkan 654 x 100 = 65400 

2. Dengan adanya model machine learning ini, e-commerce dapat dengan lebih akurat memberikan promo terhadap orang yang berpotensi Churn saja. 
- Biaya yang dibutuhkan 
    - True Positive Churn customer: 86 x 100 = 8600
    - False Positive & False Negative customer: 45 X 100 = 4500

e-commerce hanya perlu mengeluarkan biaya senilai 13100 untuk promo dalam usahanya menjaga pelanggan agar tidak churn. Hal ini dikarenakan e-commerece dapat dengan akurat mengetahui bahwa terdapat 523 pelanggan yang tidak berpotensi Churn, sehingga tidak perlu diberikan promo lagi.

Manfaat monetary yang didapatkan e-commerce dengan adanya model machine learning:
- 65400 - 13100 = 52300 --> e-commerce dapat menghemat sebesar 52300 dari biaya yang perlu dikeluarkan untuk biaya promo.

**Dampak tambahan dengan adanya model machine learning:**

Lebih lanjut, adanya machine learning memungkinkan e-commerce melakukan promo yang lebih agresif kepada pelanggan yang berpotensi churn. Harapannya adalah efektivitas promo lebih tinggi dalam menjaga mereka tetap menggunakan layanan e-commerece. --> asumsi pemberian promo 250 per pelanggan.
- True Positive Churn customer: 86 x 250 = 20750
- False Positive & False Negative customer: 45 X 250 = 11250

Total biaya yang dibutukan adalah 32000. Bahkan dengan promo yang lebih tinggi, machine learning masih mampu menghemat pengeluaran e-commerce sebesar 32650.


# **Future suggestion**

**Rekomendasi untuk e-commerce:**
1. Menggunakan model yang telah dibangun untuk mendeteksi pelanggan existing / baru terhadap kecenderungan mereka untuk Churn. Ini berguna agar e-commerce bisa menarget effortnya dengan tepat. Bagi pelanggan yang terdeteksi berpotensi Churn, maka e-commerce perlu memberikan usaha-usaha agar mereka tetap berbelanja ke platform ini. Salah satunya memberikan promo.
1. Selain bersikap reaktif (pemberian promo untuk pelanggan berpotensi Churn), e-commerce bisa melakukan serangkaian usaha lain agar pelanggan bisa loyal secara organik:
- Mempromosikan/perkenalan produk-produk lain yang ada di e-commerce => kecenderungan pelanggan churn adalah pembelian barang mobile phone yang karakteristiknya memang tidak rutin. e-commerce perlu mengenalkan produk-produk lain kepada pelanggan yang membeli barang mobile-phone agar mereka tahu bahwa mereka bisa berbelanja barang lain di platform ini. Penggunaan tools CRM menjadi salah satu opsi.
- Investasi jangka panjang untuk lokasi warehouse ==> jarak kirim dari warehouse juga menjadi faktor pelanggan churn. Lokasi warehouse yang lebih dekat diharapkan dapat mengatasi permasalahan ini. e-commerce perlu melakukan analisa tambahan dalam penentuan lokasi warehose berdasarkan lokasi pelanggan.
- Pembuatan skema loyalti ==> hal ini ditujukan untuk membuat engagement level terhadap pelanggan baru. Skema loyalti berupa pemberian promo berdasarkan keaktifan pelanggan sesuai level engagementnya bisa memotivasi pelanggan agar terus berbelanja ke platform secara organik.


**Rekomendasi perbaikan model:**

1. Eksplorasi metrik evaluasi performa lain -> salah satu opsinya adalah F2 Score (https://deepchecks.com/glossary/f-score/). Score ini berpotensi memberikan hasil evaluasi performa lebih tajam untuk kasus Churn dimana score Recall perlu bobot lebih tinggi dibandingkan Precision.
1. Menambahkan lebih banyak data khususnya untuk kelompok Churn. Hal ini agar kualitas data lebih baik dan meminimalisasi error dari penghitungan
1. Menambahkan fitur-fitur lain dari interaksi pelanggan dengan platform Churn, seperti `Last_Login`, `Total_purchase`, `Total_product_type_purchased`, `Payment_method`, dll. Fitur-fitur tersebut berpotensi untuk membuat prediksi Churn lebih akurat lagi.
1. Meningkatkan kualitas data, seperti kurangi missing value, error label, duplikat, dll

