#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:58:18 2020

@author: jasonsteffener
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

BaseDir = '/Users/jasonsteffener/Documents/GitHub/PowerMediationResults'
FileName = 'SummaryDataFile.csv'
df = pd.read_csv(os.path.join(BaseDir, FileName))

df.head()


# fig1, ((ax1, ax2, ax3, ax4, a), (ax3, ax4)) = plt.subplots(5,5)



# Plot AtoC
a = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
b = [-0, 0.1, 0.2, 0.3, 0.4, 0.5]
cP = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for k in cP:
    for i in a:
        for j in b:
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 1))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            plt.plot(df2["N"], df2["IEBCaPow"], label=str(j))
        plt.legend(title="b")
        plt.xlabel('Sample Size')
        plt.ylabel('Power of Indirect Effect')
        plt.title("a = %0.1f, c' = %0.1f"%(i,k))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,k)
        plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()
        
# # Plot AtoC
# a = [-0.2,-0.3,-0.4,-0.5]
# b = [0.2,0.3,0.4,0.5]
# cP = [0.4]
# for k in cP:
#     for i in a:
#         for j in b:
#             Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
#             N = df[Filter]["N"]
#             df2 = df[Filter]
#             df2 = df2.sort_values("N")
#             plt.plot(df2["N"], df2["IEBCaPow"], label='BCa')
#             plt.plot(df2["N"], df2["IEBCPow"], label='BC')
#             plt.plot(df2["N"], df2["IEPercPow"], label='Perc')
            
#             plt.legend(title="Confidence Intervals")
#             plt.xlabel('Sample Size')
#             plt.ylabel('Power of Indirect Effect')
#             plt.title("a = %0.1f, c' = %0.1f"%(i,k))
#             plt.xlim(0,200)
#             plt.ylim(0,1)
#             plt.hlines(0.8,0,200,linestyles='dashed')
#             plt.show()
#     # Save each figure
#         # OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f.png"%(i,k)
        # plt.savefig(os.path.join(BaseDir, OutFileName))        
        # plt.show()
        
        
            
        
        
        
        