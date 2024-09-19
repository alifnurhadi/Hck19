# Phase 1 Milestone 2

_Milestone 2 ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada Phase 1._

---

## Assignment Objectives

Milestone 2 ini dibuat guna mengevaluasi konsep Machine Learning pada pembelajaran Phase 1 sebagai berikut:

- Mampu memahami konsep Machine Learning secara keseluruhan.
- Mampu mempersiapkan data untuk digunakan dalam model Supervised Learning (Classification atau Regression).
- Mampu mengimplementasikan Supervised Learning (Classification atau Regression) dengan data yang dipilih.
- Mampu melakukan Hyperparameter Tuning dan Model Improvement.
- Mampu melakukan Model Deployment.

---

## Topik

Anda dipersilakan untuk memilih salah satu topik mengenai Supervised Learning : Regression atau Classification.

---

## Dataset

### Ketentuan Dataset
1. Pilihlah dataset yang paling nyaman digunakan karena tidak ada batasan untuk memilih dataset dalam mengerjakan Milestone 2. 

2. **Konsultasikan terlebih dahulu dataset yang hendak digunakan ke buddy masing-masing student. Jika disetujui, maka silakan dikerjakan. Jika tidak disetujui, maka cari dataset yang lain dan konsultasikan lagi mengenai dataset yang baru ini.**

3. Student tidak boleh menggunakan dataset yang sudah dipakai dalam tugas Live Code, Graded Challenge, Non Graded Challenge, dan Milestone dari Phase 0 hingga Phase 1.

4. Student juga tidak boleh menggunakan dataset yang sudah dipakai dalam sesi pembelajaran saat dikelas bersama instruktur. Carilah dataset yang baru untuk tugas Milestone 2 ini.

5. **Student dilarang untuk melakukan scraping dataset** karena dikhawatirkan proses pembuatan scraper dan proses scraping akan memakan waktu. Gunakan public dataset yang tersedia diberbagai macam situs Internet.

### Data Sources
Student dapat memilih dataset dari salah satu repository dibawah ini. Popular open data repositories :

- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon’s AWS datasets](https://registry.opendata.aws/)

Meta portals :

- [Data Portals](http://dataportals.org/)
- [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
- [Quandl](https://www.quandl.com/)
- Sumber lain yang kredibel.

---

## Conceptual Problems

*Jawab pertanyaan berikut:*

1. Jelaskan latar belakang adanya bagging dan cara kerja bagging !

2. Jelaskan perbedaan cara kerja algoritma Random Forest dengan algoritma boosting yang Anda pilih !

3. Jelaskan apa yang dimaksud dengan Cross Validation !

---

## Assignment Instructions

Milestones 2 dikerjakan dalam format ***notebook*** dan ***Model Deployment*** dengan beberapa *kriteria wajib* di bawah ini:

1. Machine learning framework yang digunakan adalah *Scikit-Learn*.

2. Ada penggunaan library visualisasi, seperti *matplotlib*, *seaborn*, atau yang lain.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, gambaran besar dataset yang digunakan, dan *objective* yang ingin dicapai.
   
   2. Import Libraries
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   3. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.
   
   4. Exploratory Data Analysis (EDA)
      > Bagian ini berisi explorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.
   
   5. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-test, transformasi data (normalisasi, encoding, dll.), dan proses-proses lain yang dibutuhkan.   
   
   6. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

   7. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
   8. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis terkait dengan hasil pada model dan tuliskan hasil analisisnya**.

   9. Model Saving
      > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model. **Dengan melihat hasil Model Evaluation, pilihlah satu model terbaik untuk disimpan. Model terbaik ini akan digunakan kembali dalam melakukan Model Inference dan Model Deployment.**
   
   10. Model Inference
       > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled. Gunakan model terbaik berdasarkan hasil Model Evaluation. Notebook Model Inference haruslah berbeda dengan notebook saat pembuatan model dilakukan.
   
   11. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan *objective* yang sudah ditulis di bagian pengenalan.

4. Notebook harus diupload dalam akun GitHub masing-masing student untuk selanjutnya dinilai.

---

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama :
  * Modeling : `P1M2_<nama-student>.ipynb`, misal `P1M2_raka_ardhi.ipynb`.
  * Model Inference : `P1M2_<nama-student>_inf.ipynb`, misal `P1M2_raka_ardhi_inf.ipynb`.

- Push Assigment yang telah Anda buat ke akun Github Classroom Anda masing-masing.

- Untuk Model Deployment :
  * Buat sebuah folder bernama `deployment` dan masukkan semua file yang berkaitan dengan deployment ke folder ini.
  * Buat sebuah file bernama `url.txt` yang berisi URL Dataset dan URL deployment.
  * Contoh bentuk isi repository dengan Model Deployment.
    ```
    ├── deployment/
    │   ├── app.py
    │   └── eda.py
    │   └── prediction.py
    │   └── model.pkl
    ├── P1M2_raka_ardhi.ipynb
    ├── P1M2_raka_ardhi_inf.ipynb
    ├── url.txt
    └── README.md
    ```
---

## Assignment Rubrics

### Code Review

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Feature Engineering | Mampu melakukan preprocessing dataset sebelum melakukan proses modeling (split data, normalisasi, encoding, dll) | 35 pts |
| KNN | Mengimplementasikan algoritma KNN pada domain kasus yang dipilih  | 5 pts |
| SVM | Mengimplementasikan algoritma SVM pada domain kasus yang dipilih  | 5 pts |
| Decision Tree | Mengimplementasikan algoritma Decision Tree pada domain kasus yang dipilih  | 5 pts |
| Random Forest | Mengimplementasikan algoritma Random Forest pada domain kasus yang dipilih  | 5 pts |
| Boosting | Mengimplementasikan salah satu algoritma Boosting pada domain kasus yang dipilih  | 5 pts |
| Pipelines | Mengimplementasikan Pipeline pada domain kasus yang dipilih | 20 pts |
| Cross Validation | Mengimplementasikan Cross Validation dengan Scikit-Learn | 25 pts |
| Hyperparameter Tuning | Mengimplementasikan Hyperparameter Tuning dengan Scikit-Learn | 20 pts |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly | Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar. | 10 pts |

```
Pada rubrik Milestone 2 diatas terdapat point Cross Validation dan Hyperparameter Tuning (GridSearchCV, RandomSearchCV, dll). 
Kedua hal yang dimaksud ini adalah dua hal yang berbeda bukan satu kesatuan. Petunjuk : 

1. Lakukan model training dengan menggunakan parameter default (baseline model) dari setiap algoritma yang diminta.
2. Kemudian, gunakan `cross_val_score` atau `cross_validate` untuk mencari nilai performansi `mean` dan `std` dari setiap model. 
3. Pilih agoritma yang terbaik dari hasil poin 2.
4. Lakukan Hyperparameter Tuning pada algoritma terbaik (berdasarkan poin 2) dengan menggunakan GridSearchCV, RandomSearchCV, dll.
5. Bandingkan performansi antara sebelum dan sesudah dilakukan Hyperparameter Tuning.
```

### Concepts

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Classifications | Mampu menjawab pertanyaan dengan singkat, jelas, dan padat serta sesuai dengan konsep dan logika yang ada mengenai Conceptual Problems (10 pts each) | 30 pts |

### Readability

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan Markdown untuk penjelasan kode | 15 pts |

```
Kriteria tertata dengan baik diantaranya adalah: 

1. Terdapat section Perkenalan yang jelas dan lengkap terkait masalah dan latar belakang masalah yang akan diselesaikan.
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll).
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 35 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan KELEBIHAN dan KELEMAHAN dari model yang dibuat DENGAN KAITANNYA DENGAN DOMAIN BUSINESS YANG DIHADAPI yang dibuktikan dengan eksplorasi sederhana (grafik, plot, teori, dll).
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Dapat menyebutkan insight yang dapat diambil setelah proses EDA, dll.
```

### Model Deployment

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Model Deployment | Membuat webapps terhadap project yang telah dibuat. | 15 pts |

```
Catatan mengenai Model Deployment : 

1. Ketiadaan URL deployment ataupun source code deployment di repository, akan tetap diperhitungkan untuk menilai bagian Model Deployment. 
2. Tidak diperkenankan adanya informasi tambahan/informasi susulan seperti lupa memberikan URL deployment atau lupa mengupload source code via apapun (DM buddy, email, atau yang lain).
3. Student akan dianggap tidak melakukan Model Deployment jika tidak ada URL deployment dan source code deployment di repository.
```

---

```
Total Points : 260

Catatan : Penilaian Milestone 2 juga dapat dipengaruhi oleh aktivitas student selama Phase 1 berlangsung, baik sesi kelas maupun sesi mentoring dengan buddy-nya masing-masing sehingga terdapat kemungkinan adanya penambahan atau pengurangan nilai diluar rubric yang telah disebutkan diatas.
```

---

## Notes

* **Deadline : P1W4D4 pukul 18:00 WIB.**

* **Keterlambatan pengumpulan tugas mengakibatkan skor Milestone 2 menjadi 0.**
