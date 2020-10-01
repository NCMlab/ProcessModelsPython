#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:42:38 2020

Load up all result files and organize them

@author: jasonsteffener
"""
import os
import csv
import pandas as pd

DataFolder = "/Users/jasonsteffener/Documents/GitHub/ProcessModelsPython/Data"
# How many result files are there?

cNames = ['Nboot','NSim','N','AtoB', 'AtoC', 'BtoC', 'typeA','powIE', 'powTE', 'powDE', 'powa', 'powb']
df = pd.DataFrame(columns=cNames)
count = 0





# Load each file
# Organize it


# Create a dataframe
# Read the data
count = 0
for filename in os.listdir(DataFolder):
    if filename.endswith(".csv"): 

       # print(os.path.join(DataFolder, filename))
        with open(os.path.join(DataFolder, filename), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        li = []
        for i in data: 
            li.append(i[0])
    
        row = pd.Series(li, index = cNames)
        
        df = df.append(row, ignore_index = True)
        count += 1
        
print(count)
df.to_csv('Data01.csv')