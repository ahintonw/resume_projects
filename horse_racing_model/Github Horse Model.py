#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math

# In[271]:


df = pd.read_csv("all_data_150.csv")

# # Remove race positions >6 and fixing indexing for race_num variables:
# indexPosition = df[(df['Finish Position'] > 6)].index
# valid_indices = indexPosition.intersection(df.index)
# df_filtered = df.drop(valid_indices)
# df_filtered = df_filtered.reset_index(drop=True)
# # new valid race num is added:
# race_num = df_filtered['Race Number']
# df = df_filtered
# reference values
ref = df['Binary Odds'].values

# Dropping columns with little data

columns_to_drop = ['Race Event Name', 'JOCK#WIN', 'JOCK%WIN', 'JIMISTDATA', 'NEWDIST']
df = df.drop(columns=columns_to_drop)
# Edit 06/17/24: for some reason prior I dropped the race number column but not I need it for grouping, it's now added bakc
race_num = df['Race Number']
# Numpy as array 

df['Odds'] = pd.to_numeric(df['Odds'], errors='coerce')
df['LIFE%WIN'] = pd.to_numeric(df['LIFE%WIN'], errors='coerce')
df['AVESPRAT'] = pd.to_numeric(df['AVESPRAT'], errors='coerce')
df['timeSinceLastRace'] = pd.to_numeric(df['timeSinceLastRace'], errors='coerce')
# Need to clean W/Race
df['W/RACE'] = df['W/RACE'].str.replace('$', '')
df['W/RACE'] = df['W/RACE'].str.replace(',', '')
df['W/RACE'] = df['W/RACE'].str.replace('.00', '')
df['W/RACE'] = pd.to_numeric(df['W/RACE'], errors='coerce')

# Need to get all values <1
df['Weight'] = df['Weight']/1000
df['Odds'] = df['Odds']/100
df['AVESPRAT'] = df['AVESPRAT']/100
df['LSPEDRAT'] = df['LSPEDRAT']/100
df['timeSinceLastRace'] = df['timeSinceLastRace']/100
df['W/RACE'] = df['W/RACE']/1000000
df['Age Of Horse'] = df['Age Of Horse']/100
b_matrix = np.matrix([
    df['Weight'].values,
    df['Start Position'].values,
    # df['Finish Position'].values,
    df['Odds'].values,
    df['LIFE%WIN'].values,
    df['AVESPRAT'].values,
    df['W/RACE'].values,
    df['LSPEDRAT'].values,
    df['timeSinceLastRace'].values,
    df['Age Of Horse'].values
])
b_matrix = b_matrix.transpose()

# dfu = pd.DataFrame(b_matrix,
#                    columns=['Weight', 'Start Position', 'Odds', 'LIFE%WIN', 'AVESPRAT', 'W/RACE',
#                             'LSPEDRAT', 'timeSinceLastRace', 'Age Of Horse'])

from sklearn.linear_model import LogisticRegression
y = np.array(ref)
x = np.array(b_matrix)

model = LogisticRegression()
model.fit(x,y)

o_w =model.coef_


df_s2 = pd.read_csv("50tsb2.csv")

# indexPosition2 = df_s2[(df_s2['Finish Position'] > 6)].index
# valid_indices2 = indexPosition2.intersection(df_s2.index)
# df_filtered2 = df_s2.drop(valid_indices2)
# df_filtered2 = df_filtered2.reset_index(drop=True)

# # new valid race num is added:
# race_num2 = df_filtered2['Race Number']
# df_s2 = df_filtered2

# reference values
ref2 = df_s2['Binary Odds'].values

# Dropping columns with little data
columns_to_drop2 = ['Race Event Name', 'JOCK#WIN', 'JOCK%WIN', 'JIMISTDATA', 'NEWDIST']
df_s2 = df_s2.drop(columns=columns_to_drop2)

# Edit 06/17/24: for some reason prior I dropped the race number column but now I need it for grouping, it's now added back
race_num2 = df_s2['Race Number']

# Numpy as array 
df_s2['Odds'] = pd.to_numeric(df_s2['Odds'], errors='coerce')
df_s2['LIFE%WIN'] = pd.to_numeric(df_s2['LIFE%WIN'], errors='coerce')
df_s2['AVESPRAT'] = pd.to_numeric(df_s2['AVESPRAT'], errors='coerce')
df_s2['timeSinceLastRace'] = pd.to_numeric(df_s2['timeSinceLastRace'], errors='coerce')

# Need to clean W/Race
df_s2['W/RACE'] = df_s2['W/RACE'].str.replace('$', '')
df_s2['W/RACE'] = df_s2['W/RACE'].str.replace(',', '')
df_s2['W/RACE'] = df_s2['W/RACE'].str.replace('.00', '')
df_s2['W/RACE'] = pd.to_numeric(df_s2['W/RACE'], errors='coerce')

# Need to get all values <1
df_s2['Weight'] = df_s2['Weight'] / 1000
df_s2['Odds'] = df_s2['Odds'] / 100
df_s2['AVESPRAT'] = df_s2['AVESPRAT'] / 100
df_s2['LSPEDRAT'] = df_s2['LSPEDRAT'] / 100
df_s2['timeSinceLastRace'] = df_s2['timeSinceLastRace'] / 100
df_s2['W/RACE'] = df_s2['W/RACE'] / 1000000
df_s2['Age Of Horse'] = df_s2['Age Of Horse'] / 100

b_matrix2 = np.asarray([
    df_s2['Weight'].values,
    df_s2['Start Position'].values,
    # df_s2['Finish Position'].values,
    df_s2['Odds'].values,
    df_s2['LIFE%WIN'].values,
    df_s2['AVESPRAT'].values,
    df_s2['W/RACE'].values,
    df_s2['LSPEDRAT'].values,
    df_s2['timeSinceLastRace'].values,
    df_s2['Age Of Horse'].values
])
b_matrix2 = b_matrix2.transpose()
ts2 = b_matrix2
ref2 = ref2

from sklearn.metrics import accuracy_score
import numpy as np

# Ensure ts2 is a valid NumPy array
ts2 = np.asarray(ts2)

# Predict probabilities for class 1 (winner)
y_pred_prob = model.predict_proba(ts2)[:, 1]

# Add the predicted probabilities to the DataFrame
df_s2['Predicted Prob'] = y_pred_prob

# Initialize a list to store the winners for each race
winners_data = []
correct_predictions = 0
total_races = 0

# Loop through each unique race and pick the horse with the highest predicted probability
for race_number in df_s2['Race Number'].unique():
    # Filter the race's horses
    race_data = df_s2[df_s2['Race Number'] == race_number]

    # Check if the race has horses
    if not race_data.empty:
        total_races += 1
        # Find the index of the horse with the highest predicted probability
        winner = race_data.loc[race_data['Predicted Prob'].idxmax()]
        winners_data.append(winner)

        # Check if the predicted winner's "Binary Odds" is 1 (correct prediction)
        if winner['Binary Odds'] == 1:
            correct_predictions += 1

# Convert the list of winners to a DataFrame
if winners_data:
    winners_df = pd.DataFrame(winners_data)

    # Show the predicted winners
    print("Predicted Winners:\n", winners_df[['Race Number', 'Horse Name', 'Predicted Prob']])

# Calculate accuracy based on correct predictions
if total_races > 0:
    accuracy = correct_predictions / total_races
    print(f"Accuracy based on Binary Odds: {accuracy:.4f}")
else:
    print("No races with valid data found.")

# Initialize a list to store matched predictions
matched_predictions = []

# Loop through each unique race and find matches
for race_number in df_s2['Race Number'].unique():
    # Filter the race's horses
    race_data = df_s2[df_s2['Race Number'] == race_number]

    # Check if the race has horses
    if not race_data.empty:
        # Find the index of the horse with the highest predicted probability
        winner = race_data.loc[race_data['Predicted Prob'].idxmax()]

        # Check if the predicted winner matches the Binary Odds winner
        if winner['Binary Odds'] == 1:
            matched_predictions.append(winner)

# Convert the matched predictions to a DataFrame
if matched_predictions:
    matched_df = pd.DataFrame(matched_predictions)
    print("Matched Predictions:\n", matched_df[['Race Number', 'Horse Name', 'Predicted Prob']])
else:
    print("No matched predictions found.")


# In[ ]:





# In[ ]:




