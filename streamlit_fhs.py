import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

df3=pd.read_csv('df3_101.csv')

from collections import defaultdict
wins_dict = defaultdict(lambda: defaultdict(int))
losses_dict = defaultdict(lambda: defaultdict(int))
draws_dict = defaultdict(lambda: defaultdict(int))

def create_dicts(df):
    for fighter, division, result in zip(df['fighter'], df['division'], df['result']):
        wins_dict[fighter][division] += result == 'W'


    for fighter, division, result in zip(df3['fighter'], df['division'], df['result']):
        losses_dict[fighter][division] += result == 'L'

    for fighter, division, result in zip(df3['fighter'], df['division'], df['result']):
        draws_dict[fighter][division] += result == 'D'
    
    return wins_dict, losses_dict, draws_dict


wins_dict, losses_dict, draws_dict = create_dicts(df3)

def result_division(df):

    df['division_wins'] = df.apply(lambda row: wins_dict[row['fighter']][row['division']], axis=1)
    df['division_losses'] = df.apply(lambda row: losses_dict[row['fighter']][row['division']], axis=1)
    df['division_draws'] = df.apply(lambda row: draws_dict[row['fighter']][row['division']], axis=1)
    df['opponent_division_wins'] = df.apply(lambda row: wins_dict[row['opponent']][row['division']], axis=1)
    df['opponent_division_losses'] = df.apply(lambda row: losses_dict[row['opponent']][row['division']], axis=1)
    df['opponent_division_draws'] = df.apply(lambda row: draws_dict[row['opponent']][row['division']], axis=1)
    return df

df3 = pd.get_dummies(df3, columns=['result'])

df3['fighter'] = df3['fighter'].astype('category')
df3['opponent'] = df3['opponent'].astype('category')
df3['division'] = df3['division'].astype('category')
df3['stance'] = df3['stance'].astype('category')
df3['opponent_stance'] = df3['opponent_stance'].astype('category')
X=df3.drop(['result_D', 'result_L', 'result_W'], axis=1)
fighters = df3['fighter'].unique()
division=df3['division'].unique()
division=np.sort(division)
#combine the two arrays into one
all_fighters=list(set(fighters))
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
    df11 = result_division(df11)
    # only keep relevant columns
    df11 = df11[X.columns]
    
    df11 = xgb.DMatrix(df11, enable_categorical=True)
    return df11


pd.set_option('display.float_format', lambda x: '%.3f' % x)
model = joblib.load('classifier_ufc.pkl')


if st.button('Predict'):
    
    df11=get_fighter_data(Fighter1, Fighter2, division)
    
    predictions = model.predict(df11)
    predictions2 = np.round(predictions).astype(int)
    if np.argmax(predictions, axis=1)==2:
        st.write('<p class="big-font">The winner is', Fighter1, 'with a probability of', "{:.2f}".format(predictions[0][2]*100), '%</p>', unsafe_allow_html=True)
    elif np.argmax(predictions, axis=1)==1:
        st.write('<p class="big-font">The winner is', Fighter2, 'with a probability of',  "{:.2f}".format(predictions[0][1]*100), '%</p>', unsafe_allow_html=True)
    elif np.argmax(predictions, axis=1)==0:
        st.write('<p class="big-font">The fight is a draw with a probability of',  "{:.2f}".format(predictions[0][0]*100), '%</p>', unsafe_allow_html=True)
    else:
        st.write('The fight is a draw')

