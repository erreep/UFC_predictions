import requests
import urllib3
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

#function for getting individual fight stats
def get_fight_stats(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    fd_columns = {'fighter':[], 'knockdowns':[],'sig_strikes':[], 'total_strikes':[], 'takedowns':[], 'sub_attempts':[], 'pass':[],
                   'reversals':[]}
    
    #gets overall fight details
    fight_details = soup.select_one('tbody.b-fight-details__table-body')
    if fight_details == None:
        print('missing fight details for:', url)
        return None
    else:
        fd_cols = fight_details.select('td.b-fight-details__table-col')
        for i in range(len(fd_cols)):
            #skip 3 and 6: strike % and takedown %, will calculate these later
            if i == 3 or i == 6:
                pass
            else:
                col = fd_cols[i].select('p')
                for row in col:
                    data = row.text.strip()
                    if i == 0: #add to fighter
                        fd_columns['fighter'].append(data)
                    elif i == 1: #add to sig strikes
                        fd_columns['knockdowns'].append(data)
                    elif i == 2: #add to total strikes
                        fd_columns['sig_strikes'].append(data)
                    elif i == 4: #add to total strikes
                        fd_columns['total_strikes'].append(data)
                    elif i == 5: # add to takedowns
                        fd_columns['takedowns'].append(data)
                    elif i == 7: # add to sub attempts
                        fd_columns['sub_attempts'].append(data)
                    elif i == 8: # add to passes
                        fd_columns['pass'].append(data)
                    elif i == 9: # add to reversals
                        fd_columns['reversals'].append(data)
        ov_details = pd.DataFrame(fd_columns)

        #get sig strike details
        sig_strike_details = soup.find('p', class_ = 'b-fight-details__collapse-link_tot',text = re.compile('Significant Strikes')).find_next('tbody', class_ = 'b-fight-details__table-body')
        sig_columns = {'fighter':[], 'head_strikes':[], 'body_strikes':[],'leg_strikes':[], 'distance_strikes':[],
                   'clinch_strikes':[], 'ground_strikes':[]}
        fd_cols = sig_strike_details.select('td.b-fight-details__table-col')
        for i in range(len(fd_cols)):
            #skip 1, 2 (sig strikes, sig %)
            if i == 1 or i == 2:
                pass
            else:
                col = fd_cols[i].select('p')
                for row in col:
                    data = row.text.strip()
                    if i == 0: #add to fighter
                        sig_columns['fighter'].append(data)
                    elif i == 3: #add to head strikes
                        sig_columns['head_strikes'].append(data)
                    elif i == 4: #add to body strikes
                        sig_columns['body_strikes'].append(data)
                    elif i == 5: #add to leg strikes
                        sig_columns['leg_strikes'].append(data)
                    elif i == 6: #add to distance strikes
                        sig_columns['distance_strikes'].append(data)
                    elif i == 7: #add to clinch strikes
                        sig_columns['clinch_strikes'].append(data)
                    elif i == 8: #add to ground strikes
                        sig_columns['ground_strikes'].append(data)
        sig_details = pd.DataFrame(sig_columns)

        cfd = pd.merge(ov_details, sig_details, on = 'fighter', how = 'left', copy = False)

        cfd['takedowns_landed'] = cfd.takedowns.str.split(' of ').str[0].astype(int)
        cfd['takedowns_attempts'] = cfd.takedowns.str.split(' of ').str[-1].astype(int)
        cfd['sig_strikes_landed'] = cfd.sig_strikes.str.split(' of ').str[0].astype(int)
        cfd['sig_strikes_attempts'] = cfd.sig_strikes.str.split(' of ').str[-1].astype(int)
        cfd['total_strikes_landed'] = cfd.total_strikes.str.split(' of ').str[0].astype(int)
        cfd['total_strikes_attempts'] = cfd.total_strikes.str.split(' of ').str[-1].astype(int)
        cfd['head_strikes_landed'] = cfd.head_strikes.str.split(' of ').str[0].astype(int)
        cfd['head_strikes_attempts'] = cfd.head_strikes.str.split(' of ').str[-1].astype(int)
        cfd['body_strikes_landed'] = cfd.body_strikes.str.split(' of ').str[0].astype(int)
        cfd['body_strikes_attempts'] = cfd.body_strikes.str.split(' of ').str[-1].astype(int)
        cfd['leg_strikes_landed'] = cfd.leg_strikes.str.split(' of ').str[0].astype(int)
        cfd['leg_strikes_attempts'] = cfd.leg_strikes.str.split(' of ').str[-1].astype(int)
        cfd['distance_strikes_landed'] = cfd.distance_strikes.str.split(' of ').str[0].astype(int)
        cfd['distance_strikes_attempts'] = cfd.distance_strikes.str.split(' of ').str[-1].astype(int)
        cfd['clinch_strikes_landed'] = cfd.clinch_strikes.str.split(' of ').str[0].astype(int)
        cfd['clinch_strikes_attempts'] = cfd.clinch_strikes.str.split(' of ').str[-1].astype(int)
        cfd['ground_strikes_landed'] = cfd.ground_strikes.str.split(' of ').str[0].astype(int)
        cfd['ground_strikes_attempts'] = cfd.ground_strikes.str.split(' of ').str[-1].astype(int)

        cfd = cfd.drop(['takedowns','sig_strikes', 'total_strikes', 'head_strikes', 'body_strikes', 'leg_strikes', 'distance_strikes', 
                        'clinch_strikes', 'ground_strikes'], axis = 1)
        return(cfd)

#function for getting fight stats for all fights on a card
def get_fight_card(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    fight_card = pd.DataFrame()
    date = soup.select_one('li.b-list__box-list-item').text.strip().split('\n')[-1].strip()
    rows = soup.select('tr.b-fight-details__table-row')[1:]
    for row in rows:
        fight_det = {'date':[], 'fight_url':[], 'event_url':[], 'result':[], 'fighter':[], 'opponent':[], 'division':[], 'method':[],
                    'round':[], 'time':[], 'fighter_url':[], 'opponent_url':[]}
        fight_det['date'] += [date, date] #add date of fight
        fight_det['event_url'] += [url, url] #add event url
        cols = row.select('td')
        for i in range(len(cols)):
            if i in set([2,3,4,5]): #skip sub, td, pass, strikes
                pass
            elif i == 0: #get fight url and results
                fight_url = cols[i].select_one('a')['href'] #get fight url
                fight_det['fight_url'] += [fight_url, fight_url]

                results = cols[i].select('p')
                if len(results) == 2: #was a draw, table shows two draws
                    fight_det['result'] += ['D', 'D']
                else: #first fighter won, second lost
                    fight_det['result'] += ['W', 'L'] 

            elif i == 1: #get fighter names and fighter urls
                fighter_1 = cols[i].select('p')[0].text.strip()
                fighter_2 = cols[i].select('p')[1].text.strip()
                
                fighter_1_url = cols[i].select('a')[0]['href']
                fighter_2_url = cols[i].select('a')[1]['href']

                fight_det['fighter'] += [fighter_1, fighter_2]
                fight_det['opponent'] += [fighter_2, fighter_1]
                
                fight_det['fighter_url'] += [fighter_1_url, fighter_2_url]
                fight_det['opponent_url'] += [fighter_2_url, fighter_1_url]
            elif i == 6: #get division
                division = cols[i].select_one('p').text.strip()
                fight_det['division'] += [division, division]
            elif i == 7: #get method
                method = cols[i].select_one('p').text.strip()
                fight_det['method'] += [method, method]
            elif i == 8: #get round
                rd = cols[i].select_one('p').text.strip()
                fight_det['round'] += [rd, rd]
            elif i == 9: #get time
                time = cols[i].select_one('p').text.strip()
                fight_det['time'] += [time, time]

        fight_det = pd.DataFrame(fight_det)
        #get striking details
        str_det = get_fight_stats(fight_url)
        if str_det is None:
            pass
        else:
            #join to fight details
            fight_det = pd.merge(fight_det, str_det, on = 'fighter', how = 'left', copy = False)
            #add fight details to fight card
            fight_card = pd.concat([fight_card, fight_det], axis = 0)  
    fight_card = fight_card.reset_index(drop = True)
    return fight_card

#function that gets stats on all fights on all cards
def get_all_fight_stats():
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser") 

    events_table = soup.select_one('tbody')
    events = [event['href'] for event in events_table.select('a')[1:]] #omit first event, future event

    fight_stats = pd.DataFrame()
    for event in events:
        print(event)
        stats = get_fight_card(event)
        fight_stats = pd.concat([fight_stats, stats], axis = 0)
        
    fight_stats = fight_stats.reset_index(drop = True)
    return fight_stats      

#gets individual fighter attributes
def get_fighter_details(fighter_urls):
    fighter_details = {'name':[], 'height':[], 'reach':[], 'stance':[], 'dob':[], 'url':[]}

    for f_url in fighter_urls:
        print(f_url)
        page = requests.get(f_url)
        soup = BeautifulSoup(page.content, "html.parser")

        fighter_name = soup.find('span', class_ = 'b-content__title-highlight').text.strip()
        fighter_details['name'].append(fighter_name)
        
        fighter_details['url'].append(f_url)

        fighter_attr = soup.find('div', class_ = 'b-list__info-box b-list__info-box_style_small-width js-guide').select('li')
        for i in range(len(fighter_attr)):
            attr = fighter_attr[i].text.split(':')[-1].strip()
            if i == 0:
                fighter_details['height'].append(attr)
            elif i == 1:
                pass #weight is always just whatever weightclass they were fighting at
            elif i == 2:
                fighter_details['reach'].append(attr)
            elif i == 3:
                fighter_details['stance'].append(attr)
            else:
                fighter_details['dob'].append(attr)
    return pd.DataFrame(fighter_details)  

#updates fight stats with newer fights
def update_fight_stats(old_stats): #takes dataframe of fight stats as input
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser") 

    events_table = soup.select_one('tbody')
    events = [event['href'] for event in events_table.select('a')[1:]] #omit first event, future event
    
    saved_events = set(old_stats.event_url.unique())
    new_stats = pd.DataFrame()
    for event in events:
        if event in saved_events:
            break
        else:
            print(event)
            stats = get_fight_card(event)
            new_stats = pd.concat([new_stats, stats], axis = 0)
    
    updated_stats = pd.concat([new_stats, old_stats], axis = 0)
    updated_stats = updated_stats.reset_index(drop = True)
    return(updated_stats)

#updates fighter attributes with new fighters not yet saved yet
def update_fighter_details(fighter_urls, saved_fighters):
    fighter_details = {'name':[], 'height':[], 'reach':[], 'stance':[], 'dob':[], 'url':[]}
    fighter_urls = set(fighter_urls)
    saved_fighter_urls = set(saved_fighters.url.unique())

    for f_url in fighter_urls:
        if f_url in saved_fighter_urls:
            pass
        else:
            print('adding new fighter:', f_url)
            page = requests.get(f_url)
            soup = BeautifulSoup(page.content, "html.parser")

            fighter_name = soup.find('span', class_ = 'b-content__title-highlight').text.strip()
            fighter_details['name'].append(fighter_name)

            fighter_details['url'].append(f_url)

            fighter_attr = soup.find('div', class_ = 'b-list__info-box b-list__info-box_style_small-width js-guide').select('li')
            for i in range(len(fighter_attr)):
                attr = fighter_attr[i].text.split(':')[-1].strip()
                if i == 0:
                    fighter_details['height'].append(attr)
                elif i == 1:
                    pass #weight is always just whatever weightclass they were fighting at
                elif i == 2:
                    fighter_details['reach'].append(attr)
                elif i == 3:
                    fighter_details['stance'].append(attr)
                else:
                    fighter_details['dob'].append(attr)
    new_fighters = pd.DataFrame(fighter_details)
    updated_fighters = pd.concat([new_fighters, saved_fighters])
    updated_fighters = updated_fighters.reset_index(drop = True)
    return updated_fighters


    suffixes = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
urls = []

fighter_dict = {}
for suffix in suffixes:
    url=f'http://ufcstats.com/statistics/fighters?char={suffix}&page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    fighter_urls = [fighter['href'] for fighter in soup.select('tbody a')]
    fighter_names = [fighter.text.strip() for fighter in soup.select('tbody a')]
    fighter_dict.update(dict(zip(fighter_names, fighter_urls)))
    
def flatten_dict(d):
    flattened = set()
    for key, value in d.items():
        if isinstance(value, list):
            flattened.update(flatten_list(value))
        elif isinstance(value, dict):
            flattened.update(flatten_dict(value))
        else:
            flattened.add(value)
    return list(flattened)

def flatten_list(l):
    flattened = set()
    for item in l:
        if isinstance(item, list):
            flattened.update(flatten_list(item))
        elif isinstance(item, dict):
            flattened.update(flatten_dict(item))
        else:
            flattened.add(item)
    return list(flattened)

flattened = flatten_dict(fighter_dict)
fighters=get_fighter_details(flattened)
fighters.to_csv('/Users/Downloads/fighter_own.csv', index = False)

df2=pd.read_csv('/Users/Downloads/fight_hist2.csv')
fighters=pd.read_csv('/Users/Downloads/fighter_own.csv')

df2=update_fight_stats(df2)
df2.to_csv('/Users/Downloads/fight_hist2.csv', index=False)


df2['round'] = pd.to_numeric(df2['round'])
#max round per fighter
df2['max_round']=df2.groupby('fighter')['round'].transform(max)
#total fights per fighter
df2['total_ufc_fights']=df2.groupby('fighter')['fighter'].transform('count')
#total wins per fighter
df2['total_ufc_wins']=df2.groupby('fighter')['result'].transform(lambda x: (x=='W').sum())
#total losses per fighter
df2['total_ufc_losses']=df2.groupby('fighter')['result'].transform(lambda x: (x=='L').sum())
#total draws per fighter
df2['total_ufc_draws']=df2.groupby('fighter')['result'].transform(lambda x: (x=='D').sum())


#total wins by KO/TKO per fighter
df2['total_ufc_ko']=df2.groupby('fighter')['method'].transform(lambda x: (x=='KO/TKO').sum())
#total wins by submission per fighter
df2['total_ufc_sub']=df2.groupby('fighter')['method'].transform(lambda x: (x=='Submission').sum())
#total wins by decision per fighter if column containd 'U-Dec' or 'S-Dec' or 'M-Dec'
df2['total_ufc_dec']=df2.groupby('fighter')['method'].transform(lambda x: (x.str.contains('U-Dec') | x.str.contains('S-Dec') | x.str.contains('M-Dec')).sum())
#total dq per fighter
df2['total_ufc_dq']=df2.groupby('fighter')['method'].transform(lambda x: (x=='DQ').sum())
#total nc per fighter
df2['total_ufc_nc']=df2.groupby('fighter')['method'].transform(lambda x: (x=='NC').sum())
#total overturned per fighter
df2['total_ufc_ovr']=df2.groupby('fighter')['method'].transform(lambda x: (x=='Overturned').sum())
#total cnc per fighter
df2['total_ufc_cnc']=df2.groupby('fighter')['method'].transform(lambda x: (x=='CNC').sum())
#total other per fighter
df2['total_ufc_other']=df2.groupby('fighter')['method'].transform(lambda x: (x=='Other').sum())


#clean division column
#remove apostrophe from division column
df2["division"] = df2["division"].str.replace("'", "")

result_count = df2.groupby(['fighter', 'division', 'result']).count()
for division in df2["division"].unique():
        df2[f'Wins_{division}'] = 0
        df2[f'Losses_{division}'] = 0
        df2[f'Draws_{division}'] = 0
for fighter in df2["fighter"].unique():
    for division in df2["division"].unique():
    #if fighter fighter is in result_count with division Bantamweight 
        if (fighter, division) in result_count.index:
            #if has wins in banweight
            if 'W' in result_count.loc[fighter, division].index:
                df2[f"Wins_{division}"][df2['fighter']==fighter] = result_count.loc[fighter, division, 'W'][0]
            if 'L' in result_count.loc[fighter, division].index:
                df2[f"Losses_{division}"][df2['fighter']==fighter] = result_count.loc[fighter, division, 'L'][0]
            if 'D' in result_count.loc[fighter, division].index:
                df2[f"Draws_{division}"][df2['fighter']==fighter] = result_count.loc[fighter, division, 'D'][0]          
        else:
            df2[f"Wins_{division}"][df2['fighter']==fighter] = 0
            df2[f"Losses_{division}"][df2['fighter']==fighter] = 0
            df2[f"Draws_{division}"][df2['fighter']==fighter] = 0


def create_fighter_stats(df, col_list):
    for col in col_list:
        df[f'total_{col}_all'] = df.groupby('fighter')[f'{col}_landed'].transform('sum')
        df[f'mean_{col}_landed_all'] = df.groupby('fighter')[f'{col}_landed'].transform('mean')
        df[f'total_{col}_attempted_all'] = df.groupby('fighter')[f'{col}_attempts'].transform('sum')
        df[f'mean_{col}_attempted_all'] = df.groupby('fighter')[f'{col}_attempts'].transform('mean')
        df[f'max_{col}_landed_all'] = df.groupby('fighter')[f'{col}_landed'].transform('max')
        df[f'min_{col}_landed_all'] = df.groupby('fighter')[f'{col}_landed'].transform('min')
        df[f'median_{col}_landed_all'] = df.groupby('fighter')[f'{col}_landed'].transform('median')
        df[f'max_{col}_attempted_all'] = df.groupby('fighter')[f'{col}_attempts'].transform('max')
        df[f'min_{col}_attempted_all'] = df.groupby('fighter')[f'{col}_attempts'].transform('min')
        df[f'median_{col}_attempted_all'] = df.groupby('fighter')[f'{col}_attempts'].transform('median')
    return df
def received_fighter_stats(df, col_list):
    for col in col_list:
        def create_dict(df, col,metric):
            result_dict = dict(zip(df['opponent'], df.groupby('opponent')[f'{col}_landed'].transform(metric)))
            return result_dict
        for i in ['sum', 'mean', 'max', 'min', 'median']:
            temp_dict = create_dict(df, col, i)
            df[f'{col}_received_{i}'] = df['fighter'].map(temp_dict)
    return df

col_list = ['clinch_strikes', 'ground_strikes', 'distance_strikes', 'head_strikes', 'body_strikes', 'total_strikes', 'sig_strikes','leg_strikes','takedowns']
df2 = received_fighter_stats(df2, col_list)
df2=create_fighter_stats(df2, col_list)


#create a dataset from df2 without the columns ['date', 'fight_url', 'event_url', 'result', 'fighter', 'opponent','division', 'method', 'round', 'time', 'fighter_url', 'opponent_url']
df3=df2.copy()
df3=df2.drop(['date', 'fight_url', 'event_url',#'division',
        'method', 'round', 'time', 'fighter_url', 'opponent_url',
       'knockdowns', 'sub_attempts', 'pass', 'reversals', 'takedowns_landed',
       'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts',
       'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed',
       'head_strikes_attempts', 'body_strikes_landed', 'body_strikes_attempts',
       'leg_strikes_landed', 'leg_strikes_attempts', 'distance_strikes_landed',
       'distance_strikes_attempts', 'clinch_strikes_landed',
       'clinch_strikes_attempts', 'ground_strikes_landed',
       'ground_strikes_attempts'], axis=1)



       #make a column stance in df3 that takes the value of stance from fighters dataset based on fighter column in df3
df3['stance']=df3['fighter'].map(dict(zip(fighters['name'], fighters['stance'])))
#height
df3['height']=df3['fighter'].map(dict(zip(fighters['name'], fighters['height'])))


import numpy as np
columns=list(df3.columns)
columns.remove('fighter')
columns.remove('opponent')
columns.remove('result')
columns.remove('division')
def add_opponent_columns(df, cols):
    for col in cols:
        opponent_col = f"opponent_{col}"
        opponent_values = df.apply(lambda x: df[df['fighter'] == x['opponent']][col].values[0] if x['opponent'] in df['fighter'].values else np.nan, axis=1)
        df[opponent_col] = opponent_values
 
    return df

new_cols=columns
df3 = add_opponent_columns(df3, new_cols)

df3.to_csv('/Users/Downloads/df3_99.csv', index=False)
#df3=pd.read_csv('/Users/sebas/Downloads/df3_99.csv')

def add_opponent_columns(df, cols):
    # create a new dataframe with the fighter and opponent columns
    opponent_values = df[['fighter', 'opponent']]
    # for each column, add a new column to the dataframe with opponent values
    for col in cols:
        opponent_col = f"opponent_{col}"
        # merge the original dataframe with the opponent values dataframe
        opponent_col_values = df.merge(opponent_values.rename(columns={'fighter': 'opponent', 'opponent': 'fighter',}), on='opponent', how='left')[col]
        df[opponent_col] = opponent_col_values
    return df

columns=list(df3.columns)
columns.remove('fighter')
columns.remove('opponent')
columns.remove('result')
new_cols=columns
df3 = add_opponent_columns(df3, new_cols)

df3['fight_url']=df2['fight_url']


#set seed
np.random.seed(0)
# define custom function to randomly select one row from each group
def select_random_row(group):
    # get value counts of result column
    value_counts = group['result'].value_counts()
    # randomly select one result value
    result_to_keep = np.random.choice(value_counts.index)
    # filter rows to keep only the randomly selected result value
    rows_to_keep = group[group['result'] == result_to_keep]
    # randomly select one row to keep
    row_to_keep = rows_to_keep.sample(n=1)
    return row_to_keep

# apply custom function to each group and combine the results
df3 = df3.groupby('fight_url').apply(select_random_row).reset_index(drop=True)
df3 = pd.get_dummies(df3, columns=['result'])
#make categorical columns
df3['fighter']=df3['fighter'].astype('category')
df3['opponent']=df3['opponent'].astype('category')
df3['stance']=df3['stance'].astype('category')
df3['opponent_stance']=df3['opponent_stance'].astype('category')

#change height to float
def convert_to_cm(height_str):
    if pd.isna(height_str) or height_str == '--':
        return float('nan')
    elif isinstance(height_str, float):
        return height_str
    else:
        feet, inch = height_str.split('\' ')
        cm = int(feet) * 30.48 + int(inch[:-1]) * 2.54
        return cm

        
df3['height'] = df3['height'].apply(convert_to_cm)
df3['opponent_height'] = df3['opponent_height'].apply(convert_to_cm)
#remove fight_url column
df3=df3.drop(['fight_url'], axis=1)

#division to category
df3['division']=df3['division'].astype('category')