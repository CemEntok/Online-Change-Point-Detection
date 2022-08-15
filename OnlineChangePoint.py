# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:57:13 2021

@author: Cem Entok

Online Change Point Detection Methods

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
# import seaborn
from scipy.stats import ks_2samp
import numba as nb

import BayesianChangePoint as oncd
from functools import partial

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('error')
    # function_raising_warning()

def generate_normal_time_series(num, minl=50, maxl=1000):
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn()*10
        var = np.random.randn()*1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return data,partition

def TraversalWindow(data,WinLen):
    
    BwWindow = np.zeros((len(data)-WinLen-1,WinLen))
    FwWindow = np.zeros((len(data)-WinLen-1,WinLen))
    for row in range(0,len(data)-WinLen-1):
        for col in range(0,WinLen):
            
            BwWindow[row,col] = data[col]
            FwWindow[row,col] = data[col+WinLen]
    return BwWindow, FwWindow

def LogLikelihoodRatio(data,winLen):
    
    # take only one batch, forward and backward window as data, i.e, data = BW+FW windows
    # LogRatio    = 0
    
    LogRatioTestStat = np.zeros(len(data))
    LH1              = np.zeros(len(data))
    LH0              = np.zeros(len(data))
    
    for kout in range(0,len(data)-2*winLen+1):
        if len(data)>=2*winLen:
            
            # Windowed Data
            data2Win = data[kout:2*winLen+kout]
            
            # H0 hypothesis testing, likelihood
            if np.var(data2Win)<0:
                breakpoint()
            sigmaH0     = np.sqrt(np.var(data2Win))
            meanH0      = np.mean(data2Win)
            LH0prev     = 1
            
            for k in range(0,len(data2Win)):
                LH0curr = np.exp(-np.square(data2Win[k]-meanH0)/(2*sigmaH0*sigmaH0))
                # if np.sum(np.isnan(LH0curr))>0:
                #     breakpoint()
                LH0curr = LH0curr*LH0prev
                LH0prev = LH0curr
            LH0curr = LH0curr * (1/np.sqrt(2*np.pi*np.power(sigmaH0,4*winLen)))
            
            # H1 hypothesis testing, likelihood
            BwWin           = data2Win[0:winLen]
            meanH1BW        = np.mean(BwWin)
            
            FwWin           = data2Win[winLen+1:-1]
            meanH1FW        = np.mean(FwWin)
            
            LH1FWprev = 1
            LH1BWprev = 1
            for k in range(0,len(FwWin)):
                LH1FWcurr = np.exp(-np.square(FwWin[k]-meanH1FW)/(2*sigmaH0*sigmaH0))
                LH1FWcurr = LH1FWcurr*LH1FWprev
                LH1FWprev = LH1FWcurr
            for k in range(0,len(BwWin)):
                LH1BWcurr = np.exp(-np.square(BwWin[k]-meanH1BW)/(2*sigmaH0*sigmaH0))
                LH1BWcurr = LH1BWcurr*LH1BWprev
                LH1BWprev = LH1BWcurr
                
            LH1curr =  (1/np.sqrt(2*np.pi*np.power(sigmaH0,4*winLen)))*LH1FWcurr*LH1BWcurr
            
            LogRatio = np.log(LH1curr/(LH0curr+np.spacing(1)))
            # if LogRatio<-10:
            #     breakpoint()
        # try:
        if kout==len(data)-2*winLen+1:
            print("a")
            breakpoint()
            
        if kout+winLen==len(LogRatioTestStat):
            print("stop")
        LogRatioTestStat[kout+winLen] = LogRatio
        # except IndexError:
        
        #     # breakpoint()
        #     print("index error")
            
        LH1[kout+winLen] = LH1curr
        LH0[kout+winLen] = LH0curr
        
    return LogRatioTestStat,LH1,LH0 
    # return LogRatio,LH1curr,LH0curr
    

def AUCStatistics(data,windowLen):
    
    # windowLen = 15;
    Stat = np.zeros(np.size(data))

    # k = 1;
    k = 0
    AUCTestStat = np.zeros(np.size(data))

    # LastInd         = 2*windowLen+k-1
    LastInd         = 2*windowLen+k
    while LastInd != len(data):

        CPindex     = k+windowLen

        BackWin     = data[k:CPindex]

        LastInd     = 2*windowLen+k

        FwWin       = data[k+windowLen:LastInd]

        for n in range(0,len(BackWin)):
            for m in range(0,len(FwWin)):

                if FwWin[m] > BackWin[n]:
                    Stat[k] = Stat[k] + 1

        AUCTestStat[CPindex] = Stat[k]/((m+1)*(n+1))

        k = k + 1

    for j in range(windowLen+1,len(AUCTestStat)-windowLen+1):
        if (AUCTestStat[j]<0.5):
            AUCTestStat[j] = 1- AUCTestStat[j]

    # plt.close("all")"
    # time = np.arange(1,len(data)+1)
    # for realInd in range(0,len(data)):
        
    # plt.close()
    # AUCplt = plt.figure()
    # axAUC = AUCplt.add_subplot(211)
    # plt.plot(time,data,MarkerSize=12)
    # # # plt.plot(time[0:realInd],data[0:realInd],MarkerSize=12)
    # plt.title('data')
    # plt.xlabel('sample')
    # plt.ylabel('value')
    # plt.grid()
    # AUCplt.add_subplot(212, sharex = axAUC)
    # plt.plot(time,AUCTestStat)
    # # # plt.plot(time[0:realInd],AUCTestStat[0:realInd])
    # plt.title('AUC Test Statistics')
    # plt.xlabel('sample');
    # plt.ylabel('Statistics')
    # plt.grid()
    # plt.pause(0.1)

    
    return AUCTestStat


def KSTest(data,winLen):
    
    
    KSTestStat = np.zeros(len(data))
    pvalKS = np.zeros(len(data))
    
    for k in range(0,len(data)-2*winLen):
        try:            
            KSTestStat[winLen+k], pvalKS[winLen+k]= ks_2samp(data[k:winLen+k],data[winLen+k:2*winLen+k])
        except ValueError:
            breakpoint()
        except:
            print("An exception occurred") 
            # breakpoint()
            
    return KSTestStat, pvalKS

def main():
    abc = np.array([0,1,2,3,4,5])
    bw,fw = TraversalWindow(abc,2)
    
    # Plot the simulated data
    data,partition = generate_normal_time_series(7, 50, 200)
    
    # data  = np.hstack((np.arange(0,1,0.02),np.arange(3,4,0.02)))

    
    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(data)
    
    ## AUC Change Point Analysis
    winLen = 15
    AUCTestStat = AUCStatistics(data,winLen)
    
    # ks_2samp    
    winLen = 15
    KSTestStat, pvalKS = KSTest(data,winLen)  
    
    figKS = plt.figure()
    axKS1 = figKS.add_subplot(3, 1, 1)
    plt.plot(data)
    plt.ylabel("Data Value")
    # plt.xlabel("Samples")
    plt.title("Simulated Data")
    figKS.add_subplot(3, 1, 2, sharex=axKS1)
    plt.plot(KSTestStat)
    plt.ylabel("Classifier")
    # plt.xlabel("Samples")
    plt.title("KS Test Statistics")
    figKS.add_subplot(3, 1, 3, sharex=axKS1)
    plt.plot(pvalKS)
    plt.ylabel("P value")
    plt.xlabel("Samples")
    plt.title("P value of KS Test Statistics")
    

    
    # Log-Likelihood Change Point Analysis
    winLen = 15
    LogRatioTestStat = LogLikelihoodRatio(data,winLen)[0]
    
    # LogRatioTestStat = np.zeros(len(data)-2*winLen+1)
    # LH1              = np.zeros(len(data)-2*winLen+1)
    # LH0              = np.zeros(len(data)-2*winLen+1)
    
    # for k in range(0,len(data)-2*winLen):
    #     LogRatioTestStat[k], LH1[k], LH0[k] = LogLikelihoodRatio(data[k:2*winLen+k],winLen)

        
    # fig1, ax1 = plt.subplots(figsize=[18, 16])
    LogPlt = plt.figure()
    axLog = LogPlt.add_subplot(2, 1, 1)
    plt.plot(data)
    plt.title('Data')
    plt.xlabel('sample');
    plt.ylabel('Data value')
    plt.grid()
    LogPlt.add_subplot(2, 1, 2, sharex=axLog)
    plt.plot(LogRatioTestStat)
    plt.title('Log Likelihood Test Statistics')
    plt.xlabel('sample');
    plt.ylabel('Test Statistics')
    plt.grid()
    
    # Bayesian Change Point Analysis
    R, maxes = oncd.online_changepoint_detection(data, partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
    
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(data)
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    sparsity = 5  # only plot every fifth data for faster display
    ax.pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
          np.array(range(0, len(R[:,0]), sparsity)), 
          -np.log(R[0:-1:sparsity, 0:-1:sparsity]), 
          cmap=cm.Greys, vmin=0, vmax=30)
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    Nw=10;
    ax.plot(R[Nw,Nw:-1])
    abc = R[Nw,Nw:-1]
    print(5)
if __name__ == "__main__":
    main()


