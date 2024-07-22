## Set of functions for transforming and clustering trajectories

import scipy.stats as scs
import numpy as np
import warnings
import pandas as pd
import helperFns as mf

def extractPredictorsFromWeights(loaded_dict, nTrain = 1):

    '''
    Extracts subject ID and drift in GLM choice bias from dict generated from, for example, analysis_GenerateTrajectories

    Args:
    loaded_dict: dictionary containing PsyTrack weights generated from, for example, analysis_GenerateTrajectories
    nTrain: number of training sessions, default is just the final training session

    returns:
    df1t: pandas dataframe with ID and drift in GLM choice bias
    '''

    last = np.sum(loaded_dict['dayLength'][-nTrain:])
    psytrack_weights = loaded_dict['psytrack']['wMode']
    psytrack_names =  np.array(list(loaded_dict['psytrack']['weights'].keys()))

    bias = np.transpose(psytrack_weights[psytrack_names == 'bias'])
    b = bias[-last:]

    bias_var = np.std(b)

    df1t = {
        'mID' : loaded_dict['subject'],
        'bias_var': bias_var,
    }

    df1t = pd.DataFrame(df1t, index = [0])
    df1t = df1t.reset_index()
    df1t = df1t.drop('index', axis = 1)

    return df1t

def extractFromTestingSession(loaded_dict, nTest = 3, delay = 0):

    '''
    Extracts psychometric parameters from dict generated from, for example, getPsychFit function in fitFns.py file

    Args:
    loaded_dict: dictionary containing choice behavior and psychometric fit results generated from, for example, getPsychFit function in fitFns.py file
    nTest: number of testing sessions to analyze (default is first 3)
    delay: if interested in skipping a number of sessions to test strength of correlation across longer time gaps

    returns:
    tempS: pandas dataframe with psychometric parameters
    '''

    fitType = loaded_dict['1']['fit_method']
    sessionIDs = np.array(list(loaded_dict.keys()))
    df = []
    tmp = 0

    if delay > 0:
        sessionIDs = sessionIDs[delay:]

    if nTest > 0:
        if nTest < len(sessionIDs):
            sessionIDs = sessionIDs[0:nTest]

    accVec = []
    exAccVec = []
    for sID in sessionIDs:

        dfT = pd.DataFrame(loaded_dict[sID]['fit_params']['mean']).T
        sf = loaded_dict[sID]['behavior']['stimulus_frequency']
        choice = loaded_dict[sID]['behavior']['choice']
        cat = loaded_dict[sID]['behavior']['stimulus_category'] - 1
        
        accTemp = np.mean(choice[cat < 2] == cat[cat < 2])
        accVec.append(accTemp)

        temp1 = np.unique(sf)
        temp2 = choice[np.logical_or(sf==temp1[0],sf==temp1[-1])]
        temp3 = cat[np.logical_or(sf==temp1[0],sf==temp1[-1])]

        exAccTemp = np.mean(temp2 == temp3)
        exAccVec.append(exAccTemp)

        minF = sf.min()
        maxF = sf.max()

        un_sf = np.unique(sf)
        diff_sf = np.mean(np.diff(un_sf))
        max_slope = 1/diff_sf
        
        if fitType == 'pymc':
            normMu = (dfT['mean'][0] - minF)/(maxF - minF)
            slope = (1/dfT['mean'][1]) * (1/np.sqrt(2*np.pi))
            slope = np.min((slope, max_slope))
        elif fitType == 'bads':
            if np.size(dfT,1) > 1:
                dfT = dfT[0] #.head(1)
            normMu = (dfT[1] - minF)/(maxF - minF) 
            slope = (1/dfT[2]) * (1/np.sqrt(2*np.pi))
            slope = np.min((slope, max_slope))

        dfT.loc[len(dfT.index)] = normMu
        dfT.loc[len(dfT.index)] = slope

        if tmp == 0:
            df = dfT
            tmp = 1
        else:
            df = pd.concat([df, dfT], axis = 1, ignore_index=True)
    
    df = pd.DataFrame(df).T

    if fitType == 'pymc':
        df.columns = [*df.columns[:-1], 'mu_norm', 'slope']
    elif fitType == 'bads':
        df.columns = ['gamma','mu','sigma','lambda','mu_norm', 'slope']

    df.columns = df.columns + '_'
    
    df.reset_index()

    tempM = df.mean(axis = 0)

    tempS = pd.DataFrame(df.std(axis = 0)/np.sqrt(len(df.columns)))

    t = tempS.transpose()

    t.columns = df.columns + 's'

    tempS = t.transpose()

    tempS = pd.concat([tempM, tempS], axis = 0).transpose()
    tempS['acc'] = np.mean(accVec)
    tempS['exAcc'] = np.mean(exAccVec)
    return tempS

def smoothLearningTraces(acc_low, acc_high, nPoints = 100, smoothF = 15, smooth2 = 5, sameSize = True):       

    '''
    Subsamples and smooths category-conditional learning traces- can also be applied to, for example, reinforcement learning model weights as they evolve

    Args:
    acc_low: accuracy conditioned on low category stimuli
    acc_high: accuracy conditioned on high category stimuli
    nPoints: nPoints to subsample to
    smoothF: smoothing factor to use for interpolating traces
    smooth2: smoothing to use for subsampling
    sameSize: whether to make all traces same length

    returns:
    nPoints: actual nPoints used
    wModeRaws: raw conditional traces, concatenated
    wModeSigned: conditional traces signed based on first session preference
    wModeDiff: trace of rate of change of conditional traces
    wModeAvg: average trace of both conditional traces
    '''

    wModeRaws = []
    wModeSigned = []
    wModeDiff = []
    wModeAvg = []

    idx = np.array(range(0,len(acc_low)))
    smooth = int(np.round(len(acc_low)/smoothF,0))

    acc_low_t = pd.Series(acc_low).rolling(smooth,5).apply(lambda x : np.nanmean(x))
    acc_high_t = pd.Series(acc_high).rolling(smooth,5).apply(lambda x : np.nanmean(x))

    if not sameSize:
        nPoints = int(np.round((len(acc_low)-1)/50))

    temp_mold = np.round(np.linspace(0,len(acc_low)-1, nPoints + 1))

    wModeRaws = [] 

    startRange = int(np.round(nPoints/11))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        binnedMean,binnedIdx, *_ = scs.binned_statistic(idx, 1 - acc_low_t, statistic=np.nanmean, bins = temp_mold, range=(idx.min(),idx.max()))
        binnedMean1 = pd.Series(binnedMean).rolling(smooth2,1).apply(lambda x : np.nanmean(x))
        diff1 = np.hstack(((0), np.diff(binnedMean1)))

        binnedMean,binnedIdx, *_ = scs.binned_statistic(idx, acc_high_t, statistic=np.nanmean, bins = temp_mold, range=(idx.min(),idx.max()))
        binnedMean = pd.Series(binnedMean).rolling(smooth2,1).apply(lambda x : np.nanmean(x))
        diff2 = np.hstack(((0), np.diff(binnedMean)))
    
    st = np.mean(np.hstack((binnedMean1[0:startRange],binnedMean[0:startRange])))

    wModeDiff.extend(diff1)
    wModeDiff.extend(diff2)
    wModeDiff.extend((diff1 + diff2)/2)
    wModeDiff = np.array(wModeDiff)   
    
    wModeRaws.extend(binnedMean1)
    wModeRaws.extend(binnedMean)
    wModeRaws.extend((binnedMean1 + binnedMean)/2)    
    wModeRaws = np.array(wModeRaws)

    if st < 0.5:
        wModeSigned.extend(1 - binnedMean)
        wModeSigned.extend(1 - binnedMean1)

        wModeAvg.extend(1 - (binnedMean1 + binnedMean)/2)
    else:
        wModeSigned.extend(binnedMean1)
        wModeSigned.extend(binnedMean)
        
        wModeAvg.extend((binnedMean1 + binnedMean)/2)

    wModeSigned = np.array(wModeSigned)

    return nPoints, wModeRaws, wModeSigned, wModeDiff, wModeAvg

def generateTraceMats(IDs, dataBase, smoothF = 15, nPoints = 100, smooth2 = 5, sizeQ = True):

    '''
    More applicable code to generate signed and average conditional traces directly from files, including intermediate steps

    Args:
    IDs: mouse IDs to include
    dataBase: what folder saved trajectory dictionaries are in (will be data/Trajectories/with_bias_learning)
    nPoints: nPoints to subsample to
    smoothF: smoothing factor to use for interpolating traces
    smooth2: smoothing to use for subsampling
    sizeQ: whether to make all traces same length

    returns:
    signedTraceMat: conditional traces signed based on first session preference
    avgTraceMat: average trace of both conditional traces
    '''

    rawTraceMat = np.full([len(IDs),nPoints*3], np.nan)
    signedTraceMat = np.full([len(IDs),nPoints*2], np.nan)
    diffTraceMat = rawTraceMat.copy()

    avgTraceMat = np.full([len(IDs),nPoints], np.nan)

    for idi, ID in enumerate(IDs):

        loaded_dict = mf.loadSavedFits(ID, dataBase, ending = '_trainingDataBias')

        acc = np.array(loaded_dict['correct'])
        cat = np.array(loaded_dict['answer'])
        emptyMat = np.empty((1,np.size(acc)))
        emptyMat[:] = np.nan
        emptyMat = emptyMat.squeeze()
        acc_low = emptyMat.copy()
        acc_high = emptyMat.copy()
        acc_low[cat == 1] = acc[cat == 1]
        acc_high[cat == 2] = acc[cat == 2]

        nPointsT, wModeRaws, wModeSigned, wModeDiff, wModeAvg = smoothLearningTraces(acc_low, acc_high, nPoints = nPoints, smoothF = smoothF, smooth2 = smooth2, sameSize = sizeQ)

        if np.size(wModeRaws) > np.size(rawTraceMat,1):
            adn = np.full([np.size(rawTraceMat,0),np.size(wModeRaws) - np.size(rawTraceMat,1)], np.nan)
            
            rawTraceMat = np.hstack([rawTraceMat, adn])
            signedTraceMat = np.hstack([signedTraceMat, adn])
            diffTraceMat = np.hstack([diffTraceMat, adn])
            avgTraceMat = np.hstack([avgTraceMat, adn])

        elif np.size(wModeRaws) < np.size(rawTraceMat,1):
            adn = np.squeeze(np.full([1, np.size(rawTraceMat,1) - np.size(wModeRaws)], np.nan))

            wModeRaws = np.hstack([wModeRaws, adn])
            wModeSigned = np.hstack([wModeSigned, adn])
            wModeDiff = np.hstack([wModeDiff, adn])
            wModeAvg = np.hstack([wModeAvg, adn])

        rawTraceMat[idi,:] = wModeRaws
        signedTraceMat[idi,:] = wModeSigned
        diffTraceMat[idi,:] = wModeDiff
        avgTraceMat[idi,:] = wModeAvg

    signedTraceMat = pd.DataFrame(signedTraceMat, index = IDs)
    avgTraceMat = pd.DataFrame(avgTraceMat, index = IDs)

    return signedTraceMat, avgTraceMat