#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:42:38 2020

Load up all result files and organize them

Something like this will make it all a lot faster

with open("temp/merged_pure_python2.csv","wb") as fout:
    # first file:
    with open("temp/in/1.csv", "rb") as f:
        fout.write(f.read())
    # now the rest:    
    for num in range(2,101):
        with open("temp/in/"+str(num)+".csv", "rb") as f:
            next(f) # skip the header
            fout.write(f.read())
            
            
@author: jasonsteffener
"""
import os
import csv
import pandas as pd
import numpy as np


def main():
    #DataFolder = "/Users/jasonsteffener/Documents/GitHub/ProcessModelsPython"
    DataFolder = "/home/steffejr/Data002/out"


# Create an array of column names
    columnNames = []
    columnNames.append('SampleSize')
    columnNames.append('NBoot')
    columnNames.append('a1')
    columnNames.append('a2')
    columnNames.append('a3')
    columnNames.append('b1')
    columnNames.append('b2')
    columnNames.append('c1P')
    columnNames.append('c2P')
    columnNames.append('c3P')
    NB = 5
    NC = 3
    SimMeans = [1, 1, 1, 1]
    # Standard deviatiosnin the simulated data
    SimStds = [1, 1, 1, 1]
    ModRange = SimMeans[3] + np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])*SimStds[3]
   
    for i in range(NB):
        columnNames.append('paramA%d'%(i+1))
    for i in range(NC):
        columnNames.append('paramB%d'%(i+1))
    for i in ModRange:
        columnNames.append('modDir_D%0.1f'%(i))
    for i in ModRange:
        columnNames.append('modInd_D%0.1f'%(i))        
    for i in ModRange:
        columnNames.append('modTot_D%0.1f'%(i))        

    # How many result files are there?
    df = pd.DataFrame(columns=columnNames)
    count = 0
    # Read the data
    count = 0
    
    for filename in os.listdir(DataFolder):
        # if filename.endswith(".csv"): 
        if filename.endswith(".csv") and filename.startswith('SimData'):
            print(os.path.join(DataFolder, filename))
            with open(os.path.join(DataFolder, filename), newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            li = []
            for i in data: 
                li.append(i[0])

            row = pd.Series(li, index = columnNames)
        
            df = df.append(row, ignore_index = True)
            count += 1
        
    df.to_csv("SummaryDataFile.csv")
if __name__ == "__main__":
    main()
