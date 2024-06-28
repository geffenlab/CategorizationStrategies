
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as scs
from scipy.optimize import minimize # finding optimal params in models
from scipy import stats             # statistical tools
from scipy.signal import butter,filtfilt

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

seed = 1000
from tslearn.clustering import TimeSeriesKMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helperFns as mf

# Clustering Functions

# First, Dimensionality Reduction: Reduce the traces to the first 10 principle components.

def pca_kmeans(X, nPoints, n_components = 10, k = 3, plot = True):

    pca = PCA(n_components)

    # So, we produce a new matrix, with each learning trace defined by it's 10 principle components. We've gone from ~ 30,000 dimensions, to 1500 dimensions, to 10 dimensions.

    X_pca = pca.fit_transform(X)

    # How much of the variance do these components contain? Together they contain more than 98%.

    print(pca.explained_variance_ratio_.cumsum())

    Sum_of_squared_distances = []
    var_score = []
    db_score = []
    sil_score = []

    K = 1
    kmeans = KMeans(n_clusters = K, random_state=0, n_init="auto")
    km = kmeans.fit(X_pca)
    x_pred = kmeans.fit_predict(X_pca)
    Sum_of_squared_distances.append(km.inertia_)
    sil_score.append(np.nan)
    var_score.append(np.nan)
    db_score.append(np.nan)

    K = range(2,8)#np.size(X,0)-1)
    for ki in K:
        kmeans = KMeans(n_clusters = ki, random_state=0, n_init="auto")
        km = kmeans.fit(X_pca)
        x_pred = kmeans.fit_predict(X_pca)
        Sum_of_squared_distances.append(km.inertia_)
        var_score.append(calinski_harabasz_score(X_pca, x_pred))
        db_score.append(davies_bouldin_score(X_pca, x_pred))
        sil_score.append(silhouette_score(X_pca, x_pred))

    #Sum_of_squared_distances = np.hstack(((np.nan),(np.diff(Sum_of_squared_distances))))
    
    #Sum_of_squared_distances = 
    
    K = range(1,8) #np.size(X,0)-1)

    if plot:
            
        fig, axs = plt.subplots(1,k + 1, figsize=(24 ,4))

        axs[0].plot(K, Sum_of_squared_distances, 'bx-', zorder = 1)
        axs[0].set_xlabel('Number of Clusters')
        axs[0].set_ylabel('Sum of Squared Distances')
        axs[0].set_title('Elbow Method For Optimal K')

        axs[0].plot(k,Sum_of_squared_distances[k-1], 'ro', zorder = 5)

        ax3 = axs[0].twinx() 
        ax3.plot(K, sil_score, 'mx-', zorder = 1)
        ax3.plot(k,sil_score[k-1], 'ro', zorder = 5)

        #ax2 = axs[0].twinx() 
        #ax2.plot(K, var_score, 'gx-', zorder = 1)
        #ax2.plot(k,var_score[k-1], 'ro', zorder = 5)


# Some code from AlindGupta at GeeksForGeeks
    kmeans = KMeans(n_clusters = k, random_state=0, n_init="auto")
    x_pred = kmeans.fit_predict(X_pca)

    if plot:

        for ia, ax in enumerate(axs):
            if ia > 0:
                temp = X[x_pred == ia-1,:].T
                ax.plot(np.array(range(0,np.size(X,1))), temp, linewidth = 1)
                ax.plot(np.array(range(0,np.size(X,1))), temp.mean(1), linewidth = 5, color = 'black')
                ax.set_ylim(-4,4)
                ax.axvline(nPoints, color = 'k', linestyle = '--')

                if nPoints*2.5 < np.size(X,1):
                    ax.axvline(2*nPoints, color = 'k', linestyle = '--')
                    
                ax.set_title('Cluster ' + str(ia))

        # Bias, S_High, S_Low
        

    else:
        axs = None

    return x_pred, X_pca, axs    



## Extracting Variables From Training Sessions

# Magnitude of weights: This clearly needs to be relative to something? Or, maybe not. Since it'll be relative to the other mice
    # in the correlation.

def extractPredictorsFromWeights(loaded_dict, nTrain):

    last = np.sum(loaded_dict['dayLength'][-nTrain:])
    psytrack_weights = loaded_dict['psytrack']['wMode']
    psytrack_names =  np.array(list(loaded_dict['psytrack']['weights'].keys()))

    s_high = np.transpose(psytrack_weights[psytrack_names == 's_high'])
    s_low = np.transpose(psytrack_weights[psytrack_names == 's_low'])
    bias = np.transpose(psytrack_weights[psytrack_names == 'bias'])

    s_l = s_low[-last:]
    s_h = s_high[-last:]
    b = bias[-last:]

    varSig_l = np.std(s_l + b)
    varSig_h = np.std(s_h + b)
    
    avgVar = (varSig_h + varSig_l)/2 

    dispSig_l = varSig_l/np.mean(s_l + b)
    dispSig_h = varSig_h/np.mean(s_h + b)
    
    h_raw = np.mean(s_h)
    l_raw = np.mean(s_l)

    l_flipped =  -1 * np.mean(s_l)

    p_l = 1/(1 + np.exp(-1 *(-1 * np.mean(s_l + b))))
    p_h = 1/(1 + np.exp(-1 *(np.mean(s_h + b))))
    
    p_asym = (1 - p_l)/(1 - p_h + 1 - p_l)

    #explt = np.std(b)/(varSig_l)
    expl_l = dispSig_l

    #var_rat = np.log(varSig_h/np.abs(np.mean(s_h))) - np.log(varSig_l/np.abs(np.mean(s_l)))
    var_rat = np.log(varSig_h) - np.log(varSig_l)

    #var_rat = np.log(varSig_h/np.abs(np.mean(s_h + b))) - np.log(varSig_l/np.abs(np.mean(s_l + b)))

    #explt = np.std(b)/(varSig_h)
    expl_h = dispSig_h

    wL = -1 * np.mean(s_l + b)
    wH = np.mean(s_h + b)

    bias_x = np.mean(b)
    bias_var = np.std(b)

    #asym = np.log(np.max((wH, 0.0001))) - np.log(np.max((-1*wL, 0.0001)))
    #abs_asym = np.abs(asym)

    asym = (wH)/(wH+wL)
  
    df1t = {
        'mID' : loaded_dict['subject'],
        'low_combined': wL,
        'high_combined': wH,
        'minW': np.min((wL, wH)),
        'maxW': np.max((wL, wH)),   
        'minS': np.min((-1*l_raw, h_raw)),
        'maxS': np.max((-1*l_raw, h_raw)),            
        'avgW': (wL + wH)/2,
        'asymmetry': asym,
        'raw_asym': (h_raw + l_raw)/2,
        'expl_l': expl_l,
        'expl_h': expl_h,
        'bias': bias_x,
        'bias_var': bias_var,
        's_low': l_raw,
        's_high': h_raw,
        'var_rat': var_rat,
        'avgVar': varSig_h,
        'p_l': p_l,
        'p_h': p_h,
        'avgP': 1 - (p_l + p_h)/2,
        'minP': 1 - np.min((p_l, p_h)),
        'p_asym': p_asym,
    }

    df1t = pd.DataFrame(df1t, index = [0])
    df1t = df1t.reset_index()
    df1t = df1t.drop('index', axis = 1)

    return df1t



# Combine Testing Data Sessions

def extractFromTestingSession(loaded_dict, nTest, delay = 0, summary = True):

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
            normMu = (dfT[1] - minF)/(maxF - minF) #(dfT[0][1] - minF)/(maxF - minF)  
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




######

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Smoothing Raw Conditional Accuracies Into Learning Curves

def smoothLearningTraces(acc_low, acc_high, nPoints = 100, smoothF = 15, smooth2 = 5, sameSize = True):       
    
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

    wModeRaws = [] #np.zeros((2,len(temp_mold) - 1))

    startRange = int(np.round(nPoints/11)) #9

    #cutoff = 1
    #fs = 300
    #order = 2
    
    #acc_low_t_f = butter_lowpass_filter(acc_low_t[~np.isnan(acc_low_t)], cutoff, fs, order)
    #acc_high_t_f = butter_lowpass_filter(acc_high_t[~np.isnan(acc_high_t)], cutoff, fs, order)

    #acc_low_t[~np.isnan(acc_low_t)] = acc_low_t_f
    #acc_high_t[~np.isnan(acc_high_t)] = acc_high_t_f

    binnedMean,binnedIdx, *_ = scs.binned_statistic(idx, 1 - acc_low_t, statistic=np.nanmean, bins = temp_mold, range=(idx.min(),idx.max()))
    binnedMean1 = pd.Series(binnedMean).rolling(smooth2,1).apply(lambda x : np.nanmean(x))
    diff1 = np.hstack(((0), np.diff(binnedMean1)))

    binnedMean,binnedIdx, *_ = scs.binned_statistic(idx, acc_high_t, statistic=np.nanmean, bins = temp_mold, range=(idx.min(),idx.max()))
    binnedMean = pd.Series(binnedMean).rolling(smooth2,1).apply(lambda x : np.nanmean(x))
    diff2 = np.hstack(((0), np.diff(binnedMean)))
    
    st = np.mean(np.hstack((binnedMean1[0:startRange],binnedMean[0:startRange])))
    #st_mid = np.mean(np.hstack((binnedMean1[startRange*3:startRange*4],binnedMean[startRange*3:startRange*4])))

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
        #wModeSigned.extend(1 - (binnedMean1 + binnedMean)/2)  

        wModeAvg.extend(1 - (binnedMean1 + binnedMean)/2)
        switch = 1
    else:
        wModeSigned.extend(binnedMean1)
        wModeSigned.extend(binnedMean)
        #wModeSigned.extend((binnedMean1 + binnedMean)/2)
        
        wModeAvg.extend((binnedMean1 + binnedMean)/2)
        switch = 0

    wModeSigned = np.array(wModeSigned)

    return nPoints, wModeRaws, wModeSigned, wModeDiff, wModeAvg #, switch

