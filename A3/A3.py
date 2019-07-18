#https://github.com/hmlanden/TrafikiPy

import pandas as pd
#import geopandas as gpd
import numpy as np
from datetime import datetime
from copy import copy

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

def Kulczynski (A,B, bin_data): #TODO definities voor meerdere ants+conseq aanpassen
#1/2(P(A|B)+P(B|A))    
    A = list(A)[0]
    B = list(B)[0]
    
    Pab = bin_data.groupby(A)[B].value_counts()/bin_data.groupby(A)[B].count()
    Pba = bin_data.groupby(B)[A].value_counts()/bin_data.groupby(B)[A].count()
    
    return 0.5*(Pab[1][1] + Pba[1][1])

def IR (supA, supB, sup):
#Imbalance Ratio
    return np.divide(abs(supA-supB), (supA+supB-sup))    


#---------------------------------------Read data-----------------------------------------
"""
cols =  ['Accident_Index','Location_Easting_OSGR','Location_Northing_OSGR','Longitude','Latitude','Police_Force','Accident_Severity','Number_of_Vehicles','Number_of_Casualties','Date','Day_of_Week','Time','Local_Authority_(District)','Local_Authority_(Highway)','1st_Road_Class,1st_Road_Number','Road_Type','Speed_limit','Junction_Detail','Junction_Control','2nd_Road_Class','2nd_Road_Number','Pedestrian_Crossing-Human_Control','Pedestrian_Crossing-Physical_Facilities','Light_Conditions','Weather_Conditions','Road_Surface_Conditions','Special_Conditions_at_Site','Carriageway_Hazards','Urban_or_Rural_Area','Did_Police_Officer_Attend_Scene_of_Accident','LSOA_of_Accident_Location','Year']

dtype = ['U38', '<f8', '<f8', '<f8', '<f8', '<f8', '<i8', '<i8', '<i8', 'U8', '<i8', 'U8', '<f8', 'U8', '<i8', '<i8', 'U38', '<i8', 'U38', 'U38', '<f8', '<f8', 'U38', 'U38', 'U38', 'U38', 'U38', 'U38', 'U38', '<i8', 'U38', 'U38', '<i8']

dicttype = {cols[i] : dtype[i] for i in range(len(cols))}
"""

acc0 = pd.read_csv('accidents_2005_to_2007.csv', dtype=None)
acc1 = pd.read_csv('accidents_2009_to_2011.csv', dtype=None)
acc2 = pd.read_csv('accidents_2012_to_2014.csv', dtype=None)

#districts = gpd.read_file('Local_Authority_Districts_Dec_2016.geojson')
#areas = gpd.read_file('Areas.shp') #TODO dit leest niet in

#traffic = pd.read_csv('ukTrafficAADF.csv', dtype='unicode')


#---------------------------------------Preprocessing-----------------------------------------
#merge all years together, TODO: check of alles goed is gegaan!!
frames = [acc0, acc1, acc2]
acc = pd.concat(frames, ignore_index=True)

#remove blank columns
acc.dropna(axis=1, how='all', inplace=True) 

#remove rows with '', 'None' or 'Unknown'
acc['Junction_Control'].replace(np.nan, 'None', inplace=True) #checken waar dit allemaal moet
acc.replace('', np.nan, inplace=True)
acc.replace('Unknown', np.nan, inplace=True)

acc.dropna(axis=0, inplace=True)

#add columns for month, day and hour
acc['Date'] = pd.to_datetime(acc['Date'], format='%d/%m/%Y')
acc['Month'] = acc['Date'].dt.month
acc['Day'] = acc['Date'].dt.day 
acc['Hour'] = pd.to_datetime(acc['Time'], format='%H:%M').dt.hour #TODO minuten meenemen?

#remove columns that are not important or with too many different string values
acc = acc.drop(['Accident_Index', 'Date', 'Time', 'Local_Authority_(Highway)','LSOA_of_Accident_Location', 'Police_Force', 'Local_Authority_(District)'  ], axis=1)

#TO DO volgende columns evt toevoegen door bins te maken
acc = acc.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude','Latitude', '1st_Road_Number', '2nd_Road_Number'], axis=1)

   
#inspect data
uniquestracc = []
uniquecols = []
for i, col in enumerate(acc.columns):
    uniquecols += [str(col)+'|'+ str(unicol) for unicol in acc[col].unique()]
    

#---------------------------------------Convert to binary array-----------------------------------------

#split numerical columns and string columns
numacc = acc.select_dtypes(['int64','float64'])
stracc = acc.select_dtypes(['object'])

bin_data = pd.DataFrame([])
for unicol in uniquecols:
    col, uni = unicol.split('|')[0], unicol.split('|')[1]
    if col in stracc:
        bin_data_col = (acc[col] == uni)
    if col in numacc:
        bin_data_col = (acc[col] == float(uni))
    bin_data = pd.concat([bin_data, bin_data_col], axis=1)
    
bin_data.columns = uniquecols


#---------------------------------------Statistics-----------------------------------------
stat = np.empty((len(numacc.columns), 5)) #mean, median, minacc, maxacc, std
for i, col in enumerate(numacc.columns):
    stat[i,0] = numacc[col].mean()
    stat[i,1] = numacc[col].median()
    stat[i,2] = numacc[col].min()
    stat[i,3] = numacc[col].max()
    stat[i,4] = numacc[col].std()

#---------------------------------------frequent itemsets-----------------------------------------
freq_itemsets = apriori(bin_data, min_support=0.2, use_colnames=True)
#---------------------------------------Experiment filters-----------------------------------------
#apply these here before we start mining association rules-----------------------------------------
#---------------------------------------Experiment filters-----------------------------------------
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.8)
rules.head()


'''
If Kulczynski is near 0 or 1, then we have an interesting rule that is negatively or positively associated respectively. 
If Kulczynski is near 0.5, then we may or may not have an interesting rule. 
Imbalance Ratio where 0 is perfectly balanced and 1 is very skewed.
#https://stats.stackexchange.com/questions/151562/evaluating-association-rules-using-kulczynski-and-imbalance-ratio
''' 
drop = []
print('Start Loop')
for i in range(len(rules)):
    kul = Kulczynski(rules.iloc[i]['antecedents'], rules.iloc[i]['consequents'], bin_data)
    imb = IR(rules.iloc[i]['antecedent support'], rules.iloc[i]['consequent support'], rules.iloc[i]['support'])

    if ((0.15 < kul < 0.85) & (imb < 0.5)):
        drop.append(i) #drop these from the rules list
        #print('drop')
        
rules=rules.drop(drop,axis=0)

#rules.to_csv('supp_0_1.csv')

# =============================================================================
#with open('supp_0_1.tex', 'w') as tf:
#    tf.write(rules[['antecedents', 'consequents','support']].to_latex())
# =============================================================================
