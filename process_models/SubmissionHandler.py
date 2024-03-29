#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:35:32 2020

@author: jasonsteffener

Notes:
https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/
"""
import os
import pandas as pd
import numpy as np
import csv
import time

def main():
    WaitTimeBetweenChecks = 60*60 # in seconds
    # Where to find the output files
    # OutDir = '/Users/jasonsteffener/Documents/GitHub/PowerMediationResults'
    PathToResultFiles = '/home/steffejr/Data002'
    # PathToResultFiles = '/Users/jasonsteffener/Documents/TEMP/out'
    PathToJobFiles = '/home/steffejr/Data002/jobs'
    
    # PathToJobFiles = '/Users/jasonsteffener/Documents/TEMP'
    # WHat is the submission list filename
    fileName = "SubmissionList.csv"
    EmptyQueueFlag = False
    JobsFinishedFlag = False
    while not EmptyQueueFlag or not JobsFinishedFlag:
        
        # Check the queue
        if CheckOnQueue():
            print("The queue is empty")
            EmptyQueueFlag = True
            # If the queue is empty, CheckSubmissions
            [JobsCompleted, JobsToDo] = CheckSubmissions(PathToResultFiles, fileName)
            
            if JobsCompleted == JobsToDo:
                JobsFinishedFlag = True
                print("ALL DONE!!!")
                # All done
            else:
                print("Submitting jobs")
                ResubmitJobs(PathToJobFiles, PathToResultFiles, fileName)
        else:
            print("The queue is not empty")
            EmptyQueueFlag = False
        # Wait one hour before checking again
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Finished last check at: %s"%(current_time))
        time.sleep(WaitTimeBetweenChecks)
        



def CheckOnQueue():
    # Check on the queue and return if the queue is empty
    P = os.popen('sq | wc').read()
    JobsInQueue = int(P.split()[0])  
    return JobsInQueue == 1

    
def CheckSubmissions(PathToResultFiles, fileName):
    # Read the CheckSubnission list
    df = pd.read_csv(os.path.join(PathToResultFiles, fileName))
    # Cycle over the list of result files and check to see if they are found    
    count = 0 
    # List all files
    files = os.listdir(os.path.join(PathToResultFiles,'out'))
    print('Checking %d files'%(len(files)))
    # cycle over all files
    for file in files:
        # Make sure it is the correct type of file
        if file.endswith(".csv") and file.startswith('SimData'): 
            # print the name of the file to stdout
            # print(os.path.join(PathToResultFiles, file))
            # OPen the file
            with open(os.path.join(PathToResultFiles,'out', file), newline='') as f:
                # Read the file
                reader = csv.reader(f)
                data = list(reader)
                # Make a list of the data in the file
                li = []
                for i in data: 
                    li.append(i[0])
    
            # Find in the simulation list the values for this result
            # Make flags for the values
            flagN = df['N'] == int(float(li[0]))
            flagNBoot = df['NBoot'] == int(float(li[1]))
            flagp001 = df['param001'] == round(float(li[2]),2)
            flagp002 = df['param002'] == round(float(li[3]),2)
            flagp003 = df['param003'] == round(float(li[4]),2)
            flagp004 = df['param004'] == round(float(li[5]),2)
            flagp005 = df['param005'] == round(float(li[6]),2)
            flagp006 = df['param006'] == round(float(li[7]),2)
            flagp007 = df['param007'] == round(float(li[8]),2)
            flagp008 = df['param008'] == round(float(li[9]),2)
            
            
            
            mask = flagN & flagNBoot & flagp001 & flagp002 & flagp003 & flagp004 & flagp005 & flagp006 & flagp007 & flagp008
            # find the position of this value
            pos = np.flatnonzero(mask)
            # if the simulation results were found update the submission list
            # file to say it was completed
            df.loc[pos,'Completed'] = 1
            count += 1
    
    print("Jobs completed: %d, out of %d total jobs"%(count, len(df.index)))
    # update the SUbmissionList file on disk
    df.to_csv(os.path.join(PathToResultFiles, fileName))    
    return count, len(df.index)

def ResubmitJobs(PathToJobFiles, PathToResultFiles, fileName):
    # read the Submission List file
    df = pd.read_csv(os.path.join(PathToResultFiles, fileName))
    
    CountSubmitted = 0
    SubmitLimit = 700
    ListOfResubmit = []
    for index, row in df.iterrows():
        # Check to see if a row has data
        if not row['Completed'] == 1.0:
            # resubmit the batch
            CurrentSimId = row['SimID']

            # If this SimID has not already been resubmitted, then do so 
            if not CurrentSimId in ListOfResubmit:
                # Add the submitted job to the list
                ListOfResubmit.append(CurrentSimId)
                # Find the batch file
                CurrentSimId = '%06d'%(CurrentSimId)
                fileName = "submit_Process_%s"%(CurrentSimId)            
                # submit the batch job
                if CountSubmitted < SubmitLimit:
                    os.system('sbatch %s'%(os.path.join(PathToJobFiles, fileName+".sh")))
                    time.sleep(0.5)
                    CountSubmitted += 1

if __name__ == "__main__":
    main()
