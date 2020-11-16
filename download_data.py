# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:18:31 2020

@author: Shawn Leavor

Script will download all IMDB-WIKI Face data to be used in the project
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def getBirthYear(date):
    '''
    Gets the year someone was born for the data format in the .mat file

    Parameters
    ----------
    date : int
        A number in raw date formaat.

    Returns
    -------
    A date's year.

    '''
    return datetime.date.fromordinal(np.max([date - 366, 1])).year

def createDF(path, index):
    '''
    Creates a dataframe using the path to a matlab file and the index
    that needs to be used to grab the data from the matlab file. Must use the
    8 keys that are used in IMDB-WIKI database.
    
    Adds age column to show the age when the photo was taken

    Parameters
    ----------
    path : string
        Path to the file to load.
    index : string
        The index that needs to be used to grab the data.

    Returns
    -------
    df: pandas dataframe
        A dataframe of the data within the .mat file
    '''

    #Create list of keys for final dataframe
    keys=['dob',
          'photo_taken',
          'full_path',
          'gender',
          'name',
          'face_location',
          'face_score',
          'second_face_score'
          ]

    #Load matlab data
    mat = loadmat(path)
    
    #Create list of data
    data_set = [data[0] for data in mat[index][0,0]]
    
    #Make list 8 columns
    data_set = data_set[0:8]
    
    #Create dataframe
    df = pd.DataFrame(data_set, keys)
    
    #Transpose dataframe to correct format
    df = df.transpose()
    
    #Put date of birth in year format
    df['dob'] = df.dob.apply(lambda dob: getBirthYear(dob))
    
    #Create age when photo was taken column
    df['age'] = df['photo_taken'] - df['dob']
    
    #Return final dataframe
    return df

#Create dataframes of both wiki and imdb data
wiki_df = createDF('data/wiki.mat','wiki')
imdb_df = createDF('data/imdb.mat', 'imdb')

#Create final dataframe by combining wiki and imdb data where age is
# equal to or under 100 and greater than 0
df = pd.concat([wiki_df,imdb_df], ignore_index=True)
df = df[(df.age > 0) & (df.age <= 100)]

#Grab only date of birth, date photo taken, gender, age, and name
df = df[['dob', 'photo_taken', 'gender', 'age', 'name']]

#Find number of photos taken for each age
age_counts = df.age.value_counts()
age_dists = age_counts/len(df)

#Plot age against number of counts at that age
#Uncomment first line to see line graph, second line to see barplot
plt.plot(age_dists.sort_index())
#plt.bar(age_counts.index, age_counts, align='center', width=1)

#Formatting Plot
plt.xlabel('Age in Years')
plt.ylabel('Fraction of Observations')
plt.title('Distribution of Observations at Each Age from 1 to 100 in IMDB-WIKI Dataset')
plt.grid()
plt.show()

#Create cumulative distribution plot
plt.plot(np.cumsum(age_dists.sort_index()))
plt.xlabel('Age in Years')
plt.ylabel('Cumulative Distribution of Observations')
plt.title('Cumulative Distribution of Observations for Each Age in IMDB-WIKI Dataset')
plt.grid()
plt.show()

#Find number and percent of photos taken between 15 and 25 years old
df15to25 = df[(df.age >= 15) & (df.age <= 25)]
num15to25 = len(df15to25)
per15to25 = num15to25 / len(df)
print(str(num15to25) + ' of the ' + str(len(df)) +
      ' photos are taken between the age of 15 and 25 years old.')

#Find the percentage of 30 year-old males in the dataset
# In the data, 1 is male 0 is female for gender
df30male = df[(df.gender == 1) & (df.age == 30)]
per30male = len(df30male) / len(df)
print('The percentage of 30-year old males in the data set is %.2f percent' % (per30male*100))