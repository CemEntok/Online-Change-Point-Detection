# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 01:41:29 2021

@author: Cem Entok

Animation on Real Time Timeseries Data and Test Statistics Calculation of Online Change Point Algorithms

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r'D:\Documents\Downloads\ffmpeg-5.0.1-essentials_build\bin\ffmpeg.exe'
from matplotlib import animation
from OnlineChangePoint import AUCStatistics as AUCTest
from OnlineChangePoint import LogLikelihoodRatio as LogLikelihood
from OnlineChangePoint import KSTest
import BayesianChangePoint as oncd
from functools import partial
# import librosa
import scipy
# from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import os

import pickle
import time
import pandas as pd

currPath = os.getcwd()


def generate_normal_time_series(num, minl: int=50, maxl: int=1000,  seed: int=50, meanK=10,varK=1):
    np.random.seed(seed) # Predefined seed, make the generated data the same for easily comparable
    data = np.array([], dtype=np.float64)
    means = np.array([], dtype=np.float64)
    variances = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean =  np.random.randn()*meanK
        var =   np.random.randn()*varK
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        means = np.concatenate((means, np.array([mean])))
        variances = np.concatenate((variances, np.array([var])))
        data = np.concatenate((data, tdata))
    return data ,partition, means, variances


def Generate2Partitions(num, minl: int=50, maxl: int=1000,  seed: int=50, diffMean=10,ratioVar=1,test="meanTest"):
    """
    num: number of partitions
    minl: minimum partition length sampled from uniform distribution
    maxl: maximum partition length sampled from uniform distribution
    seed: seed number to be selected in order to make the generated data the same for easy comparison of methods
    """
    # np.random.seed(seed) 
    data = np.array([], dtype=np.float64)
    means = np.array([], dtype=np.float64)
    variances = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num) # Return random integers from the discrete uniform distribution 
    mean =  diffMean
    var =   ratioVar
    cnt = 1
    for p in partition:
        if cnt==2:
            mean = 0
            var  = 1
        if var < 0:
            var = var * -1
            print("VARIANCE IS NEGATIVE")
            breakpoint()
        tdata       =   np.random.normal(mean, var, p)
        means       =   np.concatenate((means, np.array([mean])))
        variances   =   np.concatenate((variances, np.array([var])))
        data        =   np.concatenate((data, tdata))
        cnt = 2
    return data ,partition, means, variances

# USAGE of Generate2Partitions()
# data,partition, means, variances = Generate2Partitions(2, winLen*100,winLen*120,
                                                           # diffMean = diffMean, ratioVar = ratioVar)
#%%

# FLAGS
oneRunFlag      = 1                # If one run is investigated, turn on this. Do not forget arranging winLen, mean and var
animFlag        = 0                # Animation flag, to see animations, to save animations etc.
# dataLenMultp    = 1              # 1 is min 50 length, 2 is min 100 length of data generation
dataUsed        = 'sim'            # 'real': real motionSense data;  'sim': simulated gaussian dist. random timeseries
golayFlag       = 0                # Golay Filter Flag, 0 is off 1 is on, golay filter on input only Flag:2
realDataType    = "gzVec"          # Real time data type, acceleration, rotation etc, output of sensors
plottingFlag    = 1
excelFlag       = 1
pickleFlag      = 1
marginErr       = 5
epochs          = 100
simTestFlag     = "meanTest" # varTest or meanTest, mentioned in the Final Report detailed
extremeCaseName = ""
break_out_flag = False
if dataUsed == "sim":
    excelName       = extremeCaseName + simTestFlag + "_epoch" + str(epochs) + "_"
else:
    excelName       = extremeCaseName + "REALDATA_epoch" + str(epochs) + "_"

runCount = 0
if  dataUsed=="sim": 

    # SIMULATED DATA PARAMETERS
    stWin       = 20
    endWin      = 41
    incWin      = 10
    winLenArr   = np.arange(stWin,endWin,incWin)
    # winLenArr = np.array([20,30,50],dtype=int)
    
    methodArr = np.array([4],dtype=int) # choose a change point algorithm, 1 AUC, 2 LogLikelihood, 3 KS, 4 Bayesian (multiple choice possible to run in loop)
        
    # winLenArr = np.array([30, 10, 25],dtype=int)
    # winLenArr = np.delete(winLenArr,2)
    
    stRatio     = 1
    endRatio    = 5
    incRatio    = 0.5
    if simTestFlag=="meanTest":
        meanDiff    = np.arange(stRatio,endRatio+incRatio,incRatio)
        varRatio    = np.array([1])
    elif simTestFlag=="varTest":
        meanDiff    = np.array([0])
        varRatio    = np.arange(stRatio,endRatio+incRatio,incRatio)
    count       = -1
    
    if oneRunFlag == 0:
        
        # !!!! DEBUG - Preferably usage | comment on/off
        # winLenArr   = np.array([30])
        # meanDiff  = np.array([1])
        # varRatio  = np.array([1])
                        
        # Proper code starts here

        shapeMat        = len(winLenArr)*len(meanDiff)*len(varRatio)
        
        simParam        = np.zeros((shapeMat,3))
        truePos         = np.zeros((shapeMat,epochs))
        falsePos        = np.zeros((shapeMat,epochs))               
        falseNeg        = np.zeros((shapeMat,epochs))
        precision       = np.zeros((shapeMat,epochs))
        recall          = np.zeros((shapeMat,epochs))
        f1Score         = np.zeros((shapeMat,epochs))
        areaOfOverlap   = np.zeros((shapeMat,epochs))
        delayArr        = np.zeros((shapeMat,epochs))
    else: # ONE RUN SIMULATED DATA
                
        truePos, falsePos, falseNeg, precision, recall, f1Score, areaOfOverlap, shapeMat, simParam,delayArr   = (0,)*10
        methodArr = np.array([3])
        winLenArr   = np.array([20])
        meanDiff  = np.array([0])
        varRatio  = np.array([5])
        
        # winLenArr   = np.array([winLenArr[np.random.randint(len(winLenArr))]])
        # meanDiff   = np.array([meanDiff[np.random.randint(len(meanDiff))]])
        # varRatio    = np.array([varRatio[np.random.randint(len(varRatio))]])
        
elif dataUsed=="real":
    
    # REAL DATA PARAMETERS
    
    stWin, endWin, incWin, stRatio, endRatio, incRatio  = "","","","","",""    
    # methodArr = np.array([1,3,4],dtype=int)   
    # winLenArr = np.array([35, 30, 35],dtype=int)
    
    #!!! ONE METHOD IN REAL DATA CALCULATIONS
    methodArr = np.array([4],dtype=int)    # choose change point algorithm, 1 AUC, 2 LogLikelihood, 3 KS, 4 Bayesian
    winLenArr = np.array([30],dtype=int)
        
    stRatio,incRatio,incWin       = "","",""

    count       = -1   
    

    a1,a2 = "",""
    for k in range(len(currPath)):
        if a2+a1+currPath[k] == 'One':
            initStr = currPath[:k-2]
            break
        else:
            if k>1:
                a1 = currPath[k]
                a2 = currPath[k-1]
            else:
                continue
    path1 = (initStr + 'OneDrive - TUNI.fi\TAU_Courses\Research\Codes\\motionDataset')
    filename = "dataAlldec10v3"
    with open(path1 + "\\" + filename + ".pkl", "rb") as f:
        dataAll = pickle.load(f)
    
    dataKeys = np.asarray(list(dataAll.keys()))
    cploc = dataAll["cploc"]
    for m in range(len(dataKeys)):  # to remove cploc from dictionary keys for precise loop operations
        tmp = dataKeys[m]
        if tmp == 'cploc':
            indDel = m
            break
    
    # DELETE CPLOC FROM DICTIONARY AND ITS KEY SINCE IT IS A VIRTUAL FEATURE TO BE USED EXPLICITLY
    del dataAll['cploc']    
    dataKeys =  np.delete(dataKeys,indDel)

    numofPeople = cploc.shape[1]
    shapeMat        = len(winLenArr)*len(dataKeys)*numofPeople
    simParam        = np.zeros((len(dataKeys),numofPeople))
    truePos         = np.zeros((len(dataKeys),numofPeople))
    falsePos        = np.zeros((len(dataKeys),numofPeople))               
    falseNeg        = np.zeros((len(dataKeys),numofPeople))
    precision       = np.zeros((len(dataKeys),numofPeople))
    recall          = np.zeros((len(dataKeys),numofPeople))
    f1Score         = np.zeros((len(dataKeys),numofPeople))    
    areaOfOverlap   = np.zeros((len(dataKeys),numofPeople))
    
    PeopleVec   = np.zeros((len(dataKeys),numofPeople))

    
    featureName     = np.zeros((len(dataKeys),numofPeople),dtype='S8')
    featureName     = featureName.astype(dtype=str)
    
    meanDiff = np.array([0])
    varRatio = np.array([0])
        
for methodInd in methodArr:
    Method = methodInd  

    for winInd in winLenArr:
        winLen = winInd 
        
        methodInd += 1
        # winLen = 15     # FORCED windowlength to stop its variablity   | Comment/uncomment
        for meanInd in range(len(meanDiff)):
            
            diffMean = meanDiff[meanInd]
            if simTestFlag=="meanTest": count += 1
            for varInd in range(len(varRatio)): 
                if simTestFlag=="varTest": count += 1
                for epoch in range(epochs):
                    #%%
                    start = time.time()
                    ratioVar = varRatio[varInd]
                                                                            
                    # Generate Data #
                    # partitions are the points where true change points occur
                    if dataUsed == 'sim':
                        print("SIMULATED DATA")
                        data,partition, means, variances = Generate2Partitions(2, winLen*20,winLen*25,
                                                                                   diffMean = diffMean, ratioVar = ratioVar,test = simTestFlag) 
                        if oneRunFlag:
                            simParam = (winLen,diffMean,ratioVar)
                        else:                    
                            simParam[count,:] = winLen,diffMean,ratioVar
                        
                    elif dataUsed == 'real':
                        print("REAL DATA")
                        
                        featureNo   = np.where(dataKeys==realDataType)[0][0]
                        PeopleNo    = meanInd 
                        trials      = dataAll["trials"][:,PeopleNo]
                        
                        simParam[featureNo,PeopleNo] = winLen
                        
                        
                        featureName[featureNo,PeopleNo] = dataKeys[featureNo][0:]
                        PeopleVec[featureNo,PeopleNo] = PeopleNo
                        
                        featName = featureName[featureNo,PeopleNo][0:]                  
                                          
                        print("People No: " , PeopleNo+1)
                        print("feature No: ", featureNo)
                        
                        data = dataAll[featName][:,PeopleNo]
                        lastInd = np.where(trials==0)[0][0]
                        data = data[:lastInd]
                        trials = trials[:lastInd]
                    #%%      OCPD METHODS                                                                                                    
                    """                                     
                    ###############     OCPD METHODS      ############### 
                    """
                    # Method = 3                                    # 1: AUC
                                                                    # 2: Log- Likelihood
                                                                    # 3: Kolmogorov - Smirnov Test
                                                                    # 4: Bayesian Online Change Point Detection  
                    
                    if golayFlag == 2:
                        # Golay Filtering on input DATA
                        golayWinLen = 17 
                        golayPolyOrder = 3
                        dataold = data
                        data    = savgol_filter(data, window_length = golayWinLen, polyorder=golayPolyOrder, deriv=0) 
    
                    # """                                     
                    # # AUC ANIMATION 
                    # """
                    if Method==1:                
                        testStat = AUCTest(data,winLen)
                        methodTitle = "AUC_Stat "               
                        peakHeight = 0.8 # For meanTest
                        # peakHeight = 0.5 # For varTest
                        # if dataUsed  == 'real':
                        #     peakHeight = 0.9
                        prominence = 0.3
                        threshold  = None
                        
                        # peakHeight = 0.8
                        # prominence = 0.3
                        # threshold  = 0.001
                        
                    # """
                    # # LOG LIKELIHOOD  
                    # """
                    
                    elif Method==2:
                        testStat = LogLikelihood(data,winLen)[0]
                        methodTitle = "Log_Likelihood "
                        peakHeight = 5
                        prominence = 10
                        threshold  = 0.001
                           
                    # """
                    # # KS Test  
                    # """
                    elif Method==3:
                        testStat = KSTest(data,winLen)[0]
                        methodTitle = "Kolmogorov_Smirnov "
                        # if dataUsed  == 'real':
                        #     peakHeight = 0.8
                        # else:
                        peakHeight = 0.6
                        prominence = 0.2
                        threshold  = None
                        
                    # """
                    # # Bayesian OCPD 
                    # """ 
                    elif Method==4:
                        R, maxes = oncd.online_changepoint_detection(data, partial(oncd.constant_hazard, 250),
                                                                      oncd.StudentT(0.1, .01, 1, 0))
                        
                        
                        Nw=winLen
                        # Nw = 5 # FORCEW waited sample for Bayesian Method
                        testStat = R[Nw,Nw:-1]
                        testStat = np.append(testStat,np.zeros(len(data)-len(testStat)))
                        methodTitle = "Bayesian Online Change-Point Detection "
                        peakHeight = 0.25
                        prominence = 0.01
                        threshold  = 0.001
                        
                        # BAYESIAN PLOT FOR CHECKING ITS WORKING
                        # import matplotlib.cm as cm
                        # fig, ax = plt.subplots(figsize=[18, 16])
                        # ax = fig.add_subplot(3, 1, 1)
                        # ax.plot(data)
                        # ax = fig.add_subplot(3, 1, 2, sharex=ax)
                        # sparsity = 5  # only plot every fifth data for faster display
                        # ax.pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
                        #       np.array(range(0, len(R[:,0]), sparsity)), 
                        #       -np.log(R[0:-1:sparsity, 0:-1:sparsity]), 
                        #       cmap=cm.Greys, vmin=0, vmax=30)
                        # ax = fig.add_subplot(3, 1, 3, sharex=ax)
                        # ax.plot(R[Nw,Nw:-1])
                        
                        
                    # Time and Test statistics vector
                    time1  = np.arange(1,len(data)+1)
                    x = time1
                    y = testStat
                    del time1,testStat                    
                    
                    """
                    ##############  GOLAY FILTER SMOOTHING  #############
                    """
                    # golay filtering for test statistics smoothing, making peak finding easier. (Delay introduced, not used in final release)
                    # you should always choose an odd number for the window size and the order,
                    # of the polynomial function should always be a number lower than the window size.
                    if golayFlag==1:
                        if Method==2:
                            golayWinLen = 29
                            golayPolyOrder = 3
                            yhat    = savgol_filter(y, window_length = golayWinLen, polyorder=golayPolyOrder, deriv=0)
                        else:
                            golayWinLen = 17 
                            golayPolyOrder = 3
                            yhat    = savgol_filter(y, window_length = golayWinLen, polyorder=golayPolyOrder, deriv=0) 
                        # Note: window_length is 5 in simulated data, 15 (17 seems better) in real data! polyorder=3 ALL THE TIME
                        yFindPeak = yhat
                    elif golayFlag == 0 or golayFlag == 2:
                        yFindPeak = y
                           
                    """
                    ############# PEAK DETECTION ALGORITHM #############
                    """
                    peaks,_ = scipy.signal.find_peaks(yFindPeak, height = peakHeight, prominence = prominence,threshold = None,
                                                      wlen= winLen*2, width=None) # wlen ~30 between 40 is better
                    # wlen= winLen*2
                    # Extra Implementation for finding peaks when the method is Log-Likelihood because
                    # Its test statistics have different shape which should be analyzed like finding the maximum
                    # of a hill in a defined interval. In this interval, logarithm function has a gaussian shaped
                    # output. It has a maxima which refers to the summit of the hill. This summit is estimated
                    # change point location ofr log-likelihood test statistics.
                    if Method==2:
                        if len(peaks)>1:
                            peaksNew = np.zeros((len(peaks)-1))
                            for k in range(0,len(peaks)-1):
                                peaksNew[k] = 0.5 * peaks[k] + 0.5 * peaks[k+1]
                            peaks = peaksNew
                            peaks = peaks.astype(np.int64)                    
                        else:
                            print("lacking of loglikelihood peaks!!!!!!!!")
                    # peaks = argrelextrema(np.array(y), np.greater)[0]
                    
                    # Height:       it can be a number or an array and it is used to specify the minimal 
                    #               height that a peak should have, in order to be identified;
                    # Threshold:    is the required vertical distance between a peak and its neighboring,
                    #               very useful in the case of noisy functions where we want to avoid selecting peaks from the noise;
                    # Distance:     is the required minimal horizontal distance between neighboring peaks;
                    #               it can be really useful in cases in which we have some knowledge about the periodicity of the peaks.
                    # Prominence:   the minimum height necessary to descend to get from the summit to any higher terrain
                    
                    """    
                    #######################    EVALUATING    #######################
                    """
                    if dataUsed == 'sim':
    
                        partik = partition[:-1]                    
                        if marginErr==5:
                            gtsMatrix = np.concatenate((partik, partik +1,partik+2,partik-1,partik-2), axis=0)
                        elif marginErr == 3:
                            gtsMatrix = np.concatenate((partik, partik +1,partik-1), axis=0) 
    
                        if oneRunFlag:
                            for k in peaks:
                                if k in gtsMatrix:
                                    truePos      += 1
                            if truePos == 1:
                                falsePos     = max(len(peaks) - 1,0)
                                falseNeg     = 0
                            else:
                                falsePos     = len(peaks)
                                falseNeg     = 1
                        else:                                                   
                            if peaks.size == 0:
                                truePos[count,epoch]      = 0
                                falsePos[count,epoch]     = len(peaks)                       
                                falseNeg[count,epoch]     = 1
                            else: 
                                for k in peaks:
                                    if k in gtsMatrix:
                                        truePos[count,epoch]      = 1
                                        falsePos[count,epoch]     = len(peaks) - 1
                                        falseNeg[count,epoch]     = 0
                                        break
                                    else:
                                        truePos[count,epoch]      = 0
                                        falsePos[count,epoch]     = len(peaks)                       
                                        falseNeg[count,epoch]     = 1
                                        
                    elif dataUsed == 'real': 
    
                        partik = cploc[:,PeopleNo]                    
                        
                        # In real data, Margin of Error M = 5 is used
                        if marginErr==5:
                            gtsMatrix = np.concatenate((partik, partik +1,partik+2,partik-1,partik-2), axis=0)
                        else:
                            print('MARGIN OF ERROR CHECK')
                        gtsMatrix1 = np.reshape(gtsMatrix,(5,partik.shape[0]))
    
                        if peaks.size == 0:
                            truePos[featureNo,PeopleNo]      = 0
                            falsePos[featureNo,PeopleNo]     = len(peaks)                       
                            falseNeg[featureNo,PeopleNo]     = cploc.shape[0]
                        else: 
                            falseNeg[featureNo,PeopleNo] = cploc.shape[0]
                            for k in peaks:                            
                                if k in gtsMatrix1:
                                    truePos[featureNo,PeopleNo]      +=  1
                                    falseNeg[featureNo,PeopleNo]     += -1
                                else:
                                    falsePos[featureNo,PeopleNo]     +=  1                      
    
                                        
                    ### Parameters assigned for test stats and sample vector
                    #%%      PLOTTING
                    """    
                    #######################    PLOTTING    #######################
                    """        
                    if plottingFlag and oneRunFlag:
                        
                        tfinal = x[-1]
                        x0 = 1
                        ## Animation plotting starts from here
                        # fig,(ax01,ax02) = plt.subplots(figsize=(10,8),nrows=2, ncols=1)
                        fontsize = 15
                        if dataUsed == 'sim':
                            fig = plt.figure(figsize=(10,8))
                            ax01 =fig.add_subplot(211)
                            ax02 =fig.add_subplot(212,sharex=ax01)
                            ax01.set(xlim=(x0, tfinal), ylim=(-1+np.min(data), 1+ np.max(data)))
                            ax02.set(xlim=(x0, tfinal), ylim=(0.9*np.min(y), 1.1*np.max(y)))
                            
                            if dataUsed == 'real':
                                ax01.set_title("Real Data" + "_" + realDataType)
                            elif dataUsed == 'sim':
                                ax01.set_title("Simulated Data",fontsize = fontsize)
                            ax01.set_ylabel("Value",fontsize = fontsize)
                            # ax02.set_title(methodTitle + "Test" + " | WinLen = " + str(winLen))
                            ax02.set_title(methodTitle + "Change-Point Detection",fontsize = fontsize)
                            ax02.set_xlabel("Samples",fontsize = fontsize)
                            ax02.set_ylabel("Reflected Statistics",fontsize = fontsize)
                             
                            if golayFlag !=2:
                                ax01.plot(x,data,label='MeanDifference: '+str(meanDiff[meanInd]) + ' VarRatio: '+str(varRatio[varInd]))                                
                                ax02.plot(x,yFindPeak,marker = ".",label=' Statistics')
                            if golayFlag == 2:                        
                                ax01.plot(x,dataold,label='MeanDifference: '+str(meanDiff[meanInd]) + ' VarRatio: '+str(varRatio[varInd]))
                                ax01.plot(x,data,label='Filtered Data')  
                            
                            for i in partik:
                                i = int(i)
                                ax01.plot(x[i], data[i],marker = "X",markersize=fontsize)
                          
                            # ax02.plot(x,y,label='Test Statistics')
        
                            # for index in range(len(x)):
                                # if y[index]>peakHeight:
                                    # ax02.text(x[index], yFindPeak[index], float("{:.3f}".format(yFindPeak[index])), size=6)
                            for i in peaks:
                                ax02.plot(i+1, yFindPeak[i],marker = "X",ls="",markersize=12)
                            # ax01.legend()
                            # ax02.legend()
                            fig.tight_layout()
                            plt.show()
                            
                            
                        else: # PLOT FOR REAL DATA
                            
                            fig = plt.figure(figsize=(10,8))
                            ax01 =fig.add_subplot(311)
                            ax02 =fig.add_subplot(312,sharex=ax01)
                            ax03 =fig.add_subplot(313,sharex=ax01)
                            
                            ax01.set(xlim=(x0, tfinal), ylim=(0, 10))
                            ax02.set(xlim=(x0, tfinal), ylim=(-1+np.min(data), 1+ np.max(data)))
                            ax03.set(xlim=(x0, tfinal), ylim=(0.9*np.min(y), 1.1*np.max(y)))
                            
                            ax01.set_title("Real Data Ground Truth Change Points" + "_" + realDataType)
                            
                            ax02.set_title("Real Data" + "_" + realDataType)

                            ax02.set_ylabel("Value",fontsize = 12)
                            ax03.set_title(methodTitle)
                            # ax03.set_title(methodTitle + "Test" + " | WinLen = " + str(winLen))
                            ax03.set_xlabel("Samples",fontsize = 12)
                            ax03.set_ylabel("Reflected Statistics")
                             
                            if golayFlag !=2:
                                                                                                
                                ax01.plot(trials,label="DwnStr")
                                ax02.plot(x,data,label=realDataType)
                                ax03.plot(x,yFindPeak,marker = ".",label=' Test Statistics')
                            if golayFlag == 2:                        
                                ax01.plot(x,dataold,label='MeanDifference: '+str(meanDiff[meanInd]) + ' VarRatio: '+str(varRatio[varInd]))
                                ax01.plot(x,data,label='Filtered Data')  
                            events = ["UpStr","DwnStr","UpStr","Sit","Stand","Walk","Walk","Jog"]
                            for ind,i in enumerate(partik):
                                i = int(i)
                                ax01.plot(x[i],trials[i],marker = "X",markersize=12,label = events[ind])
                                ax02.plot(x[i], data[i],marker = "X",markersize=12)
                          
                            for i in peaks:
                                ax03.plot(i+1, yFindPeak[i],marker = "X",ls="",markersize=12)
                            ax01.legend()
                            ax02.legend()
                            ax03.legend()
                            fig.tight_layout()
                            plt.show()

                        #%% ANIMATION  ##################
            ########################################################################################################
            ########################    ANIMATION Creation          ################################################
            ########################################################################################################
                    if animFlag:
                        pointAdd = 20                                       # number of points in each step
                        fig2 = plt.figure(figsize=(10,8))
                        ax01 =fig2.add_subplot(211)
                        ax02 =fig2.add_subplot(212,sharex=ax01)
                        ax01.set(xlim=(x0, tfinal), ylim=(-1+np.min(data), 1+ np.max(data)))
                        ax02.set(xlim=(x0, tfinal), ylim=(0.9*np.min(y), 1.1*np.max(y)))
                        ax01.set_title("Real Data",fontsize = 15)
                        ax02.set_title("Bayesian Online Change-Point Detection",fontsize = 15)
                        ax01.set_ylabel("Data Value",fontsize = 15)
                        ax02.set_ylabel("Reflected Statistics",fontsize = 15)
                        ax02.set_ylabel("Samples",fontsize = 15)
                        p01 = ax01.plot(x[0],data[0])[0]
                        p02 = ax02.plot(x[0],y[0])[0]            
                        def run_animation():
                            global anim_running
                        
                            def onClick(event):
                                anim_running = True
                                if anim_running:
                                    anim.event_source.stop()
                                    anim_running = False
                                else:
                                    anim.event_source.start()
                                    anim_running = True
                        
                        # animation function.  This is called sequentially
                            # def init():
                            #     # ax01.plot(x[i], data[i],marker = "X",markersize=12)
                            #     # ax02.plot(i+1, y[i],marker = "X",markersize=12)
                            #     return p01,p02,
                                
                            def animate(i):
                                # 
                                p01.set_data(x[:i*pointAdd],data[:i*pointAdd])
                                p02.set_data(x[:i*pointAdd],y[:i*pointAdd])
                                
                                # if i in partik:
                                #     ax01.plot(x[i], data[i],marker = "X",markersize=12)
                                #     # ax01.draw_artist(x[i], data[i],marker = "X",markersize=12)
                                #     # ax01.plot(x[partik[partik==i]], data[partik[partik==i]],marker = "X",markersize=12)           
                                
                                # if i in peaks:
                                #     # p02.set_data(i+1, y[i],marker = "X",ls="",markersize=12)
                                #     ax02.plot(i+1, y[i],marker = "X",markersize=12)
                                #     # ax02.draw_artist(i+1, y[i],marker = "X",markersize=12)
                                return p01, p02,
    
                            # ax02.plot(x,yhat,marker = ".")
                            fig2.canvas.mpl_connect('button_press_event', onClick)
                            
                            anim = animation.FuncAnimation(fig2, animate, frames= data.shape[-1]//pointAdd,interval=10, repeat=False,
                                                            save_count =data.shape[-1],blit = True) # For anim.save; repeat True
                        
    
                            # Gif saving
                            # gifname = methodTitle + r"- gif.gif" 
                            # writergif = animation.PillowWriter(fps=60) 
                            # anim.save(gifname, writer=writergif)
                            
                            # Mp4 saving                   
                            videoname =  methodTitle + r"- bayes.mp4" 
                            writervideo = animation.FFMpegWriter(fps=60) 
                            anim.save(videoname, writer=writervideo)
                                                    
                            
                            # anim.save('KSgif2.gif', writer='imagemagick', fps=30)
                        
                        anim = run_animation()
                    
            ########################################################################################################
            ########################    ANIMATION End          #####################################################
            ########################################################################################################     
#%%                           
                    if dataUsed =="sim":
            
                        if oneRunFlag:
                            print("One RUN")
                            
                            print("True Positive: ", truePos)
                            print("Method: " + methodTitle)
                            print("winLen: ", winLen)
                            print("meanDiff: ", meanDiff)
                            print("varRatio: ", varRatio)
                            print("True Positives in Sim Data: ",truePos)
                            print("False Positives in Sim Data: ",falsePos)
                            print("False Negatives in Sim Data: ",falseNeg)
                            precision = truePos/(truePos+falsePos+ 1e-14)
                            print("Precision in Real Data: %.5f " %precision)
                            recall = truePos/(truePos+falseNeg+ 1e-14)
                            print("Recall in Real Data: %.5f "%recall)
                            f1Score = 2*precision*recall/(precision + recall + 1e-14)
                            print("F1-Score in Real Data: %.5f "%f1Score)
                            areaOfOverlap = truePos/(truePos + falsePos + falseNeg + 1e-14)
                            print("Area of Overlap Measure in Real Data: %.5f " %areaOfOverlap)
                                
                            if golayFlag: print("Golay Filter ON" ,
                                        "GolayWinlen: ", golayWinLen, "PolyOrder: ",golayPolyOrder)                
                            else: print("Golay Filter OFF")
                            break_out_flag = True
                            
                        else:
                            runCount +=1
                            print("Run No: %i/%i | Epoch No: %i/%i " %(runCount,(epochs)*(shapeMat),epoch+1,epochs))
                                
                    else:
                        if oneRunFlag:
                            print("One RUN")
                            print("REAL DATA PERFORMANCES...")
                            print("True Positives in Real Data: ",truePos)
                            print("False Positives in Real Data: ",falsePos)
                            print("False Negatives in Real Data: ",falseNeg)
                            precision = truePos/(truePos+falsePos+ 1e-14)
                            print("Precision in Real Data: %.5f " %precision)
                            recall = truePos/(truePos+falseNeg+ 1e-14)
                            print("Recall in Real Data: %.5f "%recall)
                            f1Score = 2*precision*recall/(precision + recall + 1e-14)
                            print("F1-Score in Real Data: %.5f "%f1Score)
                            areaOfOverlap = truePos/(truePos + falsePos + falseNeg + 1e-14)
                            # acc = truePos/(truePos + falseNeg + 1e-14)
                            print("Area of Overlap Measure in Real Data: %.5f " %areaOfOverlap)
                        else:
                            print("True Positives in Real Data: ",truePos[featureNo,PeopleNo])
                            
                    end = time.time()
                    totalTime = end - start
                    print("Method: " , methodTitle)
                    print("Time Elapsed: %.2f " %totalTime)
                    print("Estimated Remaining Time...  %.2f " %(totalTime*((epochs)*(shapeMat)-runCount)))
                    if oneRunFlag==0:
                        delayArr[count,epoch] = totalTime*1000/(len(data)-winLen*2)
                        print('Delay of calculating Statistics: %.3f [ms]'%(delayArr[count,epoch]))
                    else:
                        delayArr = totalTime*1000/(len(data)-winLen*2)
                    print('------')
                    if break_out_flag: break
    precision       = truePos/(truePos+falsePos+ 1e-14)
    precisionOld    = precision
    recall          = truePos/(truePos+falseNeg+ 1e-14)
    recallOld       = recall
    f1Score         = 2*precision*recall/(precision + recall + 1e-14) 
    f1Scoreold      = f1Score
    if oneRunFlag==0:  
        f1Score         = np.mean(f1Score,axis = 1)
        truePos         = np.mean(truePos,axis = 1) 
        falsePos        = np.mean(falsePos,axis = 1)
        falseNeg        = np.mean(falseNeg,axis = 1)  
        precision       = np.mean(precision,axis = 1) 
        recall          = np.mean(recall,axis = 1)    
        delayofMethod   = np.mean(delayArr,axis = 1)
    #%% PICKLING
    if pickleFlag and oneRunFlag==0:
        #%%
        
        # def picklingOCPD(dataUsed,values,methodTitle,simParam,meanDiff,varRatio,truePos,falsePos,
        #                               falseNeg,precision,recall,f1Score,areaOfOverlap,featureName=None,PeopleVec=None,
        #                               excelName,stWin,endWin,incWin,stRatio,endRatio,incRatio):
                    
        if dataUsed == 'sim':
            values = ["method",'simParam',"meanDiff", "varRatio", "truePos", "falsePos",
                      'falseNeg','precision','recall','f1score',"DelayofMethod"]    
            dicts = {}    
            dicts = dict(zip(values,[methodTitle,simParam,meanDiff,varRatio,truePos,falsePos,
                                      falseNeg,precision,recall,f1Score,delayofMethod]))  
        else:
            featureName[featureNo,PeopleNo] = dataKeys[featureNo][0:]
            PeopleVec[featureNo,PeopleNo] = PeopleNo
            values = ["method",'simParam','featureName','PeopleVec',"meanDiff", "varRatio", "truePos", "falsePos",
                       'falseNeg','precision','recall','f1score']    
            dicts = {}    
            dicts = dict(zip(values,[methodTitle,simParam,featureName,PeopleVec,meanDiff,varRatio,truePos,falsePos,
                                       falseNeg,precision,recall,f1Score]))     

        filename =  methodTitle[:8] + "_" + excelName + "_" + dataUsed +"_win_" + str(stWin) + "_" + str(endWin) + "_" + str(incWin) \
            + "_ratios_" + str(stRatio) + "_" + str(endRatio) + "_" + str(incRatio)
        savedir = 'May_work'
        currdir = os.getcwd()
        if not currdir[len(currdir)-len(savedir):]==savedir:
            os.chdir(".\May_work")
        # if simTestFlag=="varTest": os.chdir(".\April_work")
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(dicts, f)
    
    #%% PLOTTING
    # plt.figure()
    # plt.plot(simParam[:,0],precision,label='precision')
    # # plt.plot()
    # plt.figure()
    # plt.plot(recall)
    # plt.figure()
    # plt.plot(f1Score)
    # plt.figure()
    # plt.plot(areaOfOverlap)
        
    #%% EXCEL
    if excelFlag and oneRunFlag==0:
        #%%
        # import xlsxwriter
        
        # Create a workbook and add a worksheet.
        # workbook = xlsxwriter.Workbook('deneme.xlsx')
        # worksheet = workbook.add_worksheet()
        
        # Some data we want to write to the worksheet.
        import os
        import pickle
        # try:
        #     os.chdir(r"C:\Users\gnceen\OneDrive - TUNI.fi\TAU_Courses\Research\Codes")
        # except:
        #     os.chdir(r"D:\Documents\OneDrive - TUNI.fi\TAU_Courses\Research\Codes")
        with open(filename + ".pkl", "rb") as f:
            dataA = pickle.load(f)
        
        
        
        method = pd.Series(dataA["method"], name="method")
        
        if dataUsed=='real':
        
            try:
                falsePos = pd.Series(dataA["falsePos"], name="falsePos")
            except:
                falsePos = pd.Series(dataA["falsePos"].flatten(), name="falsePos") 
            try:       
                truePos = pd.Series(dataA["truePos"], name="truePos")
            except:
                truePos = pd.Series(dataA["truePos"].flatten(), name="truePos")
            try:       
                falseNeg = pd.Series(dataA["falseNeg"], name="falseNeg")
            except:
                falseNeg = pd.Series(dataA["falseNeg"].flatten(), name="falseNeg")
            try:       
                precision = pd.Series(dataA["precision"], name="precision")
            except:
                precision = pd.Series(dataA["precision"].flatten(), name="precision")
            try:       
                recall = pd.Series(dataA["recall"], name="recall")
            except:
                recall = pd.Series(dataA["recall"].flatten(), name="recall")
            try:       
                f1score = pd.Series(dataA["f1score"], name="f1-Score")
            except:
                f1score = pd.Series(dataA["f1score"].flatten(), name="f1-Score")
            try:       
                areaOfOverlap = pd.Series(dataA["areaOfOverlap"], name="Area of Overlap")
            except:
                areaOfOverlap = pd.Series(dataA["areaOfOverlap"].flatten(), name="Area of Overlap")
            try:       
                featureName = pd.Series(dataA["featureName"], name="Feature Name")
            except:
                featureName = pd.Series(dataA["featureName"].flatten(), name="Feature Name")     
            try:       
                PeopleVec = pd.Series(dataA["PeopleVec"], name="People No")
            except:
                PeopleVec = pd.Series(dataA["PeopleVec"].flatten(), name="People No")  
            try:       
                winlen = pd.Series(dataA["simParam"], name="Window Length")
            except:
                winlen = pd.Series(dataA["simParam"].flatten(), name="Window Length")      
                

            df = pd.concat([method,winlen,featureName,PeopleVec,truePos,falsePos,falseNeg,precision,recall,f1score,
                            areaOfOverlap], axis=1)
        
        elif dataUsed=='sim':

             try:
                 falsePos = pd.Series(dataA["falsePos"], name="falsePos")
             except:
                 falsePos = pd.Series(dataA["falsePos"].flatten(), name="falsePos") 
             try:       
                 truePos = pd.Series(dataA["truePos"], name="truePos")
             except:
                 truePos = pd.Series(dataA["truePos"].flatten(), name="truePos")
             try:       
                 falseNeg = pd.Series(dataA["falseNeg"], name="falseNeg")
             except:
                 falseNeg = pd.Series(dataA["falseNeg"].flatten(), name="falseNeg")
             try:       
                 precision = pd.Series(dataA["precision"], name="precision")
             except:
                 precision = pd.Series(dataA["precision"].flatten(), name="precision")
             try:       
                 recall = pd.Series(dataA["recall"], name="recall")
             except:
                 recall = pd.Series(dataA["recall"].flatten(), name="recall")
             try:       
                 f1score = pd.Series(dataA["f1score"], name="f1-Score")
             except:
                 f1score = pd.Series(dataA["f1score"].flatten(), name="f1-Score")
             try:       
                 DelayofMethod = pd.Series(dataA["DelayofMethod"], name="DelayofMethod")
             except:
                 DelayofMethod = pd.Series(dataA["DelayofMethod"].flatten(), name="DelayofMethod")
  
             meanDelay = pd.Series(sum(dataA["DelayofMethod"].flatten())/dataA["DelayofMethod"].shape[0], name="mean-Delay[ms]")
             meanF1 = pd.Series(sum(dataA["f1score"].flatten())/dataA["f1score"].shape[0], name="mean-F1Score")
             peakHeight = pd.Series(peakHeight, name="PeakHeightThreshold")
             winlen = pd.Series(dataA["simParam"][:,0], name="winlen")
             meanDiff = pd.Series(dataA["simParam"][:,1], name="meanDiff")
             VarRatio = pd.Series(dataA["simParam"][:,2], name="VarRatio")
             df = pd.concat([method,winlen,meanDiff,VarRatio,truePos,falsePos,falseNeg,precision,recall,f1score,DelayofMethod,
                             meanF1,peakHeight,meanDelay], axis=1)           

        filepath = filename + ".xlsx"
        
        df.to_excel(filepath, index=True)
        # precisionOld = pd.Series(precisionOld, name="precisionWrtEpoch")
        # recallOld = pd.Series(recallOld, name="recallWrtEpoch")
        # f1Scoreold = pd.Series(f1Scoreold, name="f1ScoreWrtEpoch")
        # df2 = pd.concat([precisionOld,recallOld,f1Scoreold], axis=1) 
        # filepath2 = "v2_" + filepath
        # df2.to_excel(filepath2, index=True,sheet_name="sheet2")
        # os.chdir(currPath)

