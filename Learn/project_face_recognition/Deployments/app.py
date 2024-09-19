import streamlit as st
import EDA
import Prediction

page = st.sidebar.selectbox('Pilih Halaman : ',('EDA','Prediction'))

if page == 'EDA':
    EDA.Run()

elif page == 'Prediction':
    Prediction.Run()

else:
    None
