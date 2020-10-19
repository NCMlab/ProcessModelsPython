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

def main():
    #DataFolder = "/Users/jasonsteffener/Documents/GitHub/ProcessModelsPython"
    DataFolder = "/home/steffejr/Data"


    cNames = ['NBoot', 'NSimMC', 'N', 'a', 'b', 'cP', 'typeA','PercPow','BCPow','BCaPow']
    cNames.extend(['SaMean', 'SaStd', 'SbMean', 'SbStd', 'ScPMean', 'ScPStd', 'SIEMean', 'SIEStd'])
    cNames.extend(['IEBiasMean', 'IEBiasStd', 'IEBSskewMean', 'IEBSskewStd', 'IEBSskewStatMean', 'IEBSskewStatStd'])
    # How many result files are there?
    df = pd.DataFrame(columns=cNames)
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

            row = pd.Series(li, index = cNames)
        
            df = df.append(row, ignore_index = True)
            count += 1
        
    df.to_csv("SummaryDataFile.csv")
if __name__ == "__main__":
    main()
