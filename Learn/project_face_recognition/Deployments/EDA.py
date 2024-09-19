import streamlit as st
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

IMAGE = Image.open('Bank.png')

def Run():

    # Membuat title. 

    # penulisannya mirip dengan syntax markdown notebook.
    st.title(' :sparkles: Selamat Datang :sparkles:')

    # Membuat sub-header:
    st.subheader(' Exploratory Data Analysis ')

    # Membuat sub-header:
    st.subheader('Memprediksi Customer Churn')

    # buat garis lurus. 
    st.markdown('---')

    # push an image
    st.image(IMAGE, caption= 'Bank')

    # load and show dataframe.
    df = pl.read_csv('Dataset_milestone2.csv')
    st.dataframe(df)

    # EDA 1
    st.markdown(' ##  Distribusi Usia Nasabah')
    fig = plt.figure(figsize=(12,10))
    sns.histplot(data=df, x='Age', hue='Exited', kde=True, element='step')
    plt.title('Distribusi Usia Nasabah berdasarkan Status Churn')
    plt.xlabel('Usia')
    plt.ylabel('Jumlah Nasabah')
    plt.show()
    st.pyplot(fig)
    st.write(' Insight : ')
    st.write(''' rata-rata umur nasabah yang memutuskan untuk keluar dari layanan perbankan ada direntan umur 40 hingga 50 tahun''')
    

    # EDA 2
    st.markdown(' ##  Tingkat Customer Churn tiap negara')

    fig, ax = plt.subplots(figsize=(12, 10))

    filter = df.filter(pl.col('Age') >= 17).drop('Complain').select(pl.exclude(['CustomerId', 'Surname']))
    eda4 = filter.group_by('Geography').agg([pl.col('Exited').sum()]).sort('Exited', descending=True)
    eda4_1 = eda4.to_pandas()

    ax.bar(eda4_1['Geography'], eda4_1['Exited'], color='skyblue')
    ax.set_xlabel('Geography')
    ax.set_ylabel('Total Exited')
    ax.set_title('Total Exited by Geography')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown('## Detil Statistik Deskriptif')
    eda5 = df.filter((pl.col('Geography')!='Spain')&(pl.col('Exited')==1)).select(pl.exclude(['Geography','Exited'])).select(pl.col(['Age' , 'Balance' , 'NumOfProducts' , 'IsActiveMember' ])).describe()
    eda5_1 = df.filter((pl.col('Geography')!='Spain')&(pl.col('Exited')==0)).select(pl.exclude(['Geography','Exited'])).select(pl.col(['Age' , 'Balance' , 'NumOfProducts' , 'IsActiveMember' ])).describe()
    eda5_pandas = eda5.to_pandas()
    eda5_1_pandas = eda5_1.to_pandas()

    st.subheader('Statistik Deskriptif untuk customer yang memutuskan berhenti menggunakan layanan bank')
    st.dataframe(eda5_pandas)
    st.write(' Insight : ')
    st.write(''' Umur nasabah yang memutuskan untuk berhenti menggunakan layanan bank berada direntan umur yang senior,
             dengan rata-rata  umurnya adalah 45 tahun dan tidak aktif sebagai member dari bank''')

    st.subheader('Statistik Deskriptif untuk customer yang tetap dengan layanan bank')
    st.dataframe(eda5_1_pandas)
    st.write(' Insight : ')
    st.write(''' Umur nasabah yang memutuskan untuk berhenti menggunakan layanan bank berada direntan umur yang cukup muda,
             dengan rata-rata  umurnya adalah 37 tahun dan nasabah tersebut aktif sebagai member dari bank''')
    

    # EDA 3
    st.markdown('## Tingkat Churn Berdasarkan banyaknya produk yang digunakan oleh nasabah')
    EDA = df.to_pandas()
    product_churn = EDA.groupby('NumOfProducts')['Exited'].mean().reset_index()
    fig3 = plt.figure(figsize=(10, 6))
    
    plt.bar(product_churn['NumOfProducts'], product_churn['Exited'])
    plt.title('Tingkat Churn berdasarkan Jumlah Produk')
    plt.xlabel('Jumlah Produk')
    plt.ylabel('Tingkat Churn')

    # Ensure x-axis shows integer values
    plt.xticks(product_churn['NumOfProducts'])
    # Add value labels on top of each bar
    for i, v in enumerate(product_churn['Exited']):
        plt.text(product_churn['NumOfProducts'][i], v, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    st.pyplot(fig3)

    st.write(' Insight : ')
    st.write(''' semakin tinggi jumlah produk bank yang digun akan oleh nasabah kemungkinan nasabah tersebut akan memutuskan tiggal juga meningkat.
             hal tersebut bisa diindikasikan nasabah tersebut tidak bisa memenuhi kewajiban dari setiap produk yang digunakannya.''')
    
    # EDA 4
    st.markdown('## komparasi keluar atau tetapnya nasabah berdasarkan lama menggunakan layanan bank')
    tenure = EDA.groupby('Exited')['Tenure'].value_counts().unstack().fillna(0)
    st.dataframe(tenure)

    st.write(''' Insight :
Untuk pelanggan baru (tenure 0), tingkat churn tinggi mungkin menunjukkan masalah dalam onboarding atau pengalaman awal.
             Retensi meningkat hingga 3-4 tahun tenure, 
             namun churn tertinggi terjadi pada 1-2 tahun, menunjukkan periode kritis. 
             Pelanggan jangka panjang (tenure 8-10) menunjukkan loyalitas tinggi, meski jumlahnya lebih sedikit. 
             Fokus perlu diberikan pada pengurangan churn awal dan mengubah pelanggan jangka menengah menjadi loyal.

Tanda positif: retensi baik setelah tahun-tahun awal dan loyalitas kuat di pelanggan jangka panjang.  ''')


    # EDA 5
    st.markdown('## tingkatan Churn berdasarkan grup umur dan kategori saldo nasabah')
    EDA['AgeGroup'] = pd.cut(EDA['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['0-30', '31-40', '41-50', '51-60', '60+'])

    # Create BalanceGroup with dynamic labels based on actual bins created
    balance_bins = pd.qcut(EDA['Balance'], q=5, duplicates='drop')

    # Create labels dynamically based on the number of unique bins
    balance_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'][:balance_bins.cat.categories.size]

    EDA['BalanceGroup'] = pd.qcut(EDA['Balance'], q=5, labels=balance_labels, duplicates='drop')

    # Now create the heatmap
    age_balance_churn = EDA.groupby(['AgeGroup', 'BalanceGroup'])['Exited'].mean().unstack()

    edake5 = plt.figure(figsize=(10, 8))
    sns.heatmap(age_balance_churn, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Churn Rate by Age Group and Balance Group')
    plt.show()
    st.pyplot(edake5)
    st.write('''Warna pada peta panas ini bervariasi dari kuning muda hingga merah tua, di mana warna yang lebih gelap menandakan tingkat churn yang lebih tinggi. 
             Angka di dalam setiap sel menunjukkan persentase churn yang tepat untuk kombinasi kelompok usia dan saldo tertentu. 
             Kelompok usia dibagi menjadi lima kategori: 0-30, 31-40, 41-50, 51-60, dan 60+ tahun. 
             Sementara itu, saldo rekening dikelompokkan menjadi empat kategori dari "Very Low" (Sangat Rendah) hingga "High" (Tinggi). 
             Terlihat bahwa tingkat churn tertinggi (0,63 atau 63%) terjadi pada kelompok usia 51-60 tahun dengan saldo rekening "Medium" (Menengah), 
             sedangkan tingkat churn terendah (0,04 atau 4%) terdapat pada kelompok usia 0-30 tahun dengan saldo rekening "Very Low" (Sangat Rendah).''')
    
    # EDA 6
    active_churn = EDA[EDA['IsActiveMember'] == 1]['Exited'].mean()
    inactive_churn = EDA[EDA['IsActiveMember'] == 0]['Exited'].mean()

    # Display the churn rates using Streamlit
    st.write(f"### Churn Rate Analysis")
    st.write(f"**Churn rate for active members:** {active_churn:.2%}")
    st.write(f"**Churn rate for inactive members:** {inactive_churn:.2%}")
    st.write(''' Insight: 
             Tingkat churn untuk anggota yang tidak aktif (26,87%) hampir dua kali lipat dibandingkan dengan anggota yang aktif (14,27%). 
             Perbedaan signifikan ini menyoroti pentingnya keterlibatan pelanggan dalam retensi.
            Hal tersebut bisa diselesaikan seharusnya dengan meningkatkan aktivitas dan keterlibatan anggota bisa menjadi strategi yang kuat untuk mengurangi churn.''')

    # EDA 7

    st.markdown(' ## Proporsi Gender pada kondisi nasabah keluar dan tetap dengan layanan bank')

    churned_gender_dist = EDA[EDA['Exited'] == 1]['Gender'].value_counts(normalize=True)
    retained_gender_dist = EDA[EDA['Exited'] == 0]['Gender'].value_counts(normalize=True)

    # Create a bar plot to visualize the distributions
    figfig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    churned_gender_dist.plot(kind='bar', ax=ax1, title='Nasabah yang memutuskan tidak menggunakan layanan bank')
    ax1.set_ylabel('Proportion')
    ax1.set_ylim(0, 1)

    retained_gender_dist.plot(kind='bar', ax=ax2, title='Nasabah yang tetap menggunakan layanan bank')
    ax2.set_ylabel('Proportion')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    st.pyplot(figfig)
    st.write(''' Insight : 
             Ada perbedaan yang mencolok dalam tingkat churn berdasarkan gender. 
             Sekitar 60% pelanggan yang memutuskan untuk melepas layanan dengan bank adalah perempuan, 
             sedangkan 60% pelanggan yang tetap adalah laki-laki.   ''')
if __name__ == '__main__':
    Run()