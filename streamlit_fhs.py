import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb


df3=pd.read_csv('/Users/sebas/Downloads/df3_100.csv')
df3['fighter'] = df3['fighter'].astype('category')
df3['opponent'] = df3['opponent'].astype('category')
df3['division'] = df3['division'].astype('category')
df3['stance'] = df3['stance'].astype('category')
df3['opponent_stance'] = df3['opponent_stance'].astype('category')
X=df3.drop(['result_D', 'result_L', 'result_W'], axis=1)
fighters = df3['fighter'].unique()
opponents = df3['opponent'].unique()
division=df3['division'].unique()
division=np.sort(division)
#combine the two arrays into one
all_fighters = np.concatenate((fighters, opponents), axis=0)
all_fighters=list(set(all_fighters))
all_fighters=np.sort(all_fighters)
st.write("# UFC Predictor")

Fighter1 = st.selectbox("Enter fighter 1",all_fighters)
Fighter2 = st.selectbox("Enter fighter 2",all_fighters)
division = st.selectbox("Enter the division",division)

#st.button('Predict')


def get_fighter_data(fighter1, fighter2, division):

    lijst = list(df3.columns[~df3.columns.str.startswith('opponent') & ~df3.columns.str.startswith('result')])

    lijst2 = list(df3.columns[df3.columns.str.startswith('opponent')])

    df9 = df3[(df3['fighter'] == fighter1)]

    df9 = df9[lijst]
    df9 = df9.iloc[0:1]
    df9 = df9.reset_index(drop=True)
    df9['index'] = df9.index

    # create df10 where fighter is fighter2
    df10 = df3[(df3['opponent'] == fighter2)]

    # only keep columns in lijst2
    df10 = df10[lijst2]

    # delete every row but the first
    df10 = df10.iloc[0:1]
    df10 = df10.reset_index(drop=True)
    df10['index'] = df10.index

    # join df9 and df10
    df11 = df9.merge(df10, on='index', how='left')

    # drop index column
    df11 = df11.drop(columns=['index'])

    # add division column
    df11['division'] = division
    df11['division'] = df11['division'].astype('category')

    # only keep relevant columns
    df11 = df11[X.columns]
    
    df11 = xgb.DMatrix(df11, enable_categorical=True)
    return df11



model = joblib.load('/Users/sebas/Downloads/classifier.pkl')


if st.button('Predict'):
    
    df11=get_fighter_data(Fighter1, Fighter2, division)
    
    predictions = model.predict(df11)
    predictions2 = np.round(predictions).astype(int)
    if predictions2[0][2]==1:
         st.write('<p class="big-font">The winner is.</p>', Fighter1, 'with a probability of', "{:.2f}".format(predictions[0][2]*100), '%')
    elif predictions2[0][1]==1:
        st.write('<p class="big-font">The winner is.</p>', Fighter1, 'with a probability of', "{:.2f}".format(predictions[0][1]*100), '%')
    else:
        st.write('The fight is a draw')

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
