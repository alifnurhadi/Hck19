import streamlit as st
import pandas as pd
import pickle

# load all needed files.

with open('best_model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)


def Run():
    with st.form('Silahkan isi data :'):
        CreditScore = st.number_input('Credit Score: ', value=100, min_value=0, help='Input a number between 0 and 850')
        Geography = st.selectbox('Geography: ', ('France', 'Spain', 'Germany'), index=0)
        Gender = st.selectbox('Gender: ', ('Male', 'Female'), index=1)
        Age = st.number_input('Age: ', value=17, min_value=17, max_value=100, help='Please input a number between 17 and 100')
        Tenure = st.number_input('Tenure (Years): ', value=2, min_value=0, max_value=10, help='Number of years with the bank')
        Balance = st.number_input('Balance: ', value=1000.0, min_value=0.0 , max_value=500000.0, help='Account balance')
        NumOfProducts = st.number_input('Number of Products: ', value=2, min_value=0, max_value=4, help='Number of bank products')
        HasCrCard = st.selectbox('Has Credit Card: ', (0, 1), index=1, help= ' 1 if has credit card, 0 for none')
        IsActiveMember = st.selectbox('Is Active Member: ', (0, 1), index=1 , help= ' 1 if you"re an active member, 0 for none')
        EstimatedSalary = st.number_input('Estimated Salary: ', value=15000.0, min_value=0.0, max_value=200000.0, help='Estimated salary')
        SatisfactionScore = st.selectbox('Satisfaction Score: ',(0,1,2,3,4,5), index=1, help='Satisfaction score between 1 and 5')
        CardType = st.selectbox('Card Type: ', ('SILVER', 'GOLD', 'PLATINUM'), index=0)
        PointEarned = st.number_input('Point Earned: ', value=777, min_value=0, max_value=1000, help='Loyalty points earned')
    
        # Set a submit button.
        submit_button = st.form_submit_button('Submit')

    # Handle the form submission and display the data.
    if submit_button:
        data_inf = {
            'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary,
            'Satisfaction Score': SatisfactionScore,
            'Card Type': CardType,
            'Point Earned': PointEarned
        }

        # Convert the dictionary to a DataFrame and display it.
        data_inf = pd.DataFrame([data_inf])
        
        st.dataframe(data_inf)


        pred = model.predict(data_inf)
        st.write('## Hasil: ', str(int(pred)))
        st.write(' Ket : 0 [Tetap/Stay] ;  1 [Keluar/Leave]')

if __name__ == '__main__':
    Run()

    