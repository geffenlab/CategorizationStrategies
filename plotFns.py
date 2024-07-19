## Set of functions for plotting and visualization. Elements adapted from Roy et al, 2021

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.stats as scs
import clusteringFns as clf

COLORS = {'bias' : '#FAA61A', 
          's_high' : "#A9373B", 's_low' : "#2369BD",
          'Φ(s_high + bias)' : "#A9373B", 'Φ(s_low + bias)' : "#2369BD",          
          'c' : '#59C3C3', 'h' : '#9593D9', 's_avg' : '#99CC66',
          'emp_perf': '#E32D91', 'emp_bias': '#9252AB'}
ZORDER = {'bias' : 2,         
          's_high' : 3, 's_low' : 3,
          'Φ(s_high + bias)' : 3, 'Φ(s_low + bias)' : 3,          
          'c' : 1, 'h' : 1, 's_avg' : 1}


def getColors(string):
    return COLORS[string] 

colorsList = ('green', 'red', 'blue', 'orange', 'magenta', 'yellow', 'purple')

def plot_weights(W, weight_dict=None, ax = None, colors=None, zorder=None, errorbar=None, days=None): # Adapted from Roy et al, 2021

    ax = ax or plt.gca()

    # Some useful values to have around
    K, N = W.shape
    maxval = np.nanmax(np.abs(W))*1.1  # largest magnitude of any weight
    if colors is None: colors = COLORS
    if zorder is None: zorder = ZORDER

    # Infer (alphabetical) order of weights from dict
    if weight_dict is not None:
        labels = []
        for j in sorted(weight_dict.keys()):
            labels += [j]*weight_dict[j]
    else:
        labels = [i for i in range(K)]
        colors = {i: np.unique(list(COLORS.values()))[i] for i in range(K)}
        zorder = {i: i+1 for i in range(K)}

    # Plot weights and credible intervals

    for i, w in enumerate(labels):
        ax.plot(W[i], lw=1.5, alpha=0.8, ls='-', c=colors[w],
                    zorder=zorder[w], label=w)
        if errorbar is not None:  # Plot 95% credible intervals on weights
            ax.fill_between(np.arange(N),
                                W[i]-1.96*errorbar[i], W[i]+1.96*errorbar[i], 
                                facecolor=colors[w], zorder=zorder[w], alpha=0.2)

    # Plot vertical session lines
    if days is not None:
        if type(days) not in [list, np.ndarray]:
            raise Exception('days must be a list or array.')
        if np.sum(days) <= N:  # this means day lengths were passed
            days = np.cumsum(days)
        for d in days:
            if d < N:
                ax.axvline(d, c='black', ls='-', lw=0.5, alpha=0.5, zorder=0)

    # Further tweaks to make plot nice
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    ax.set_yticks(np.arange(-int(2*maxval), int(2*maxval)+1,1))
    ax.set_ylim(-maxval, maxval); ax.set_xlim(0, N)
    ax.set_xlabel('Trial #'); ax.set_ylabel('Weights')
    
    return ax
    
def plot_performance(dat, ax = None, xval_pL=None, sigma=50): # Adapted from Roy et al, 2021

    if "correct" not in dat or "answer" not in dat:
        raise Exception("Please define a `correct` {0,1} and an `answer` {1,2} "
                        "field in `dat`.")
    
    N = len(dat['y'])
    if 2 in np.unique(dat['answer']):
        answerR = (dat['answer'] == 2).astype(float)
    else:
        answerR = (dat['answer'] == 1).astype(float)

    ### Plotting
    ax = ax or plt.gca()

    # Smoothing vector for errorbars
    QQQ = np.zeros(10001)
    QQQ[5000] = 1
    QQQ = gaussian_filter(QQQ, sigma)

    # Calculate smooth representation of binary accuracy
    raw_correct = dat['correct'].astype(float)
    smooth_correct = gaussian_filter(raw_correct, sigma)
    
    # Calculate errorbars on empirical performance
    perf_errorbars = np.sqrt(
        np.sum(QQQ**2) * gaussian_filter(
            (raw_correct - smooth_correct)**2, sigma))    
    
    ax.plot(smooth_correct, c=COLORS['emp_perf'], lw=3, zorder=4, label = "Accuracy")
    ax.fill_between(range(N),
                        smooth_correct - 1.96 * perf_errorbars,
                        smooth_correct + 1.96 * perf_errorbars,
                        facecolor=COLORS['emp_perf'], alpha=0.3, zorder=3)

    # Calculate the predicted accuracy
    if xval_pL is not None:
        pred_correct = np.abs(answerR - xval_pL)
        smooth_pred_correct = gaussian_filter(pred_correct, sigma)
        ax.plot(smooth_pred_correct, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if 'dayLength' in dat and dat['dayLength'] is not None:
        days = np.cumsum(dat['dayLength'])
        for d in days:
            ax.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
                
    # Add plotting details

    ax.axhline(0.5, c='k', ls='--', lw=1, alpha=0.5, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, N); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel('Trial #'); ax.set_ylabel('Performance')

    return ax

def plot_bias(dat, ax = None, xval_pL=None, sigma=50): # Adapted from Roy et al, 2021
    
    ax = ax or plt.gca()

    if "answer" not in dat:
        raise Exception("Please define an `answer` {1,2} field in `dat`.")
        
    N = len(dat['y'])
    if 2 in np.unique(dat['y']):
        choiceR = (dat['y'] == 2).astype(float)
    else:
        choiceR = (dat['y'] == 1).astype(float)
        
    if 2 in np.unique(dat['answer']):
        answerR = (dat['answer'] == 2).astype(float)
    else:
        answerR = (dat['answer'] == 1).astype(float)

    ### Plotting
     
    # Smoothing vector for errorbars
    QQQ = np.zeros(10001)
    QQQ[5000] = 1
    QQQ = gaussian_filter(QQQ, sigma)

    # Calculate smooth representation of empirical bias
    raw_bias = choiceR - answerR
    smooth_bias = gaussian_filter(raw_bias, sigma)
    

    # Calculate errorbars on empirical performance
    bias_errorbars = np.sqrt(
        np.sum(QQQ**2) * gaussian_filter((raw_bias - smooth_bias)**2, sigma))
    
    ax.plot(smooth_bias, c=COLORS['emp_bias'], lw=3, zorder=4, label = "Bias")
    ax.fill_between(range(N),
                        smooth_bias - 1.96 * bias_errorbars,
                        smooth_bias + 1.96 * bias_errorbars,
                        facecolor=COLORS['emp_bias'], alpha=0.3, zorder=3)
        
    ### Calculate the predicted bias
    if xval_pL is not None:
        pred_bias = (1 - xval_pL) - answerR
        smooth_pred_bias = gaussian_filter(pred_bias, sigma)
        if (ax == None):
            plt.plot(smooth_pred_bias, c='k', alpha=0.75, lw=2, zorder=6)
        else:
            ax.plot(smooth_pred_bias, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if 'dayLength' in dat and dat['dayLength'] is not None:
        days = np.cumsum(dat['dayLength'])
        for d in days:
            ax.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
                
    # Add plotting details
    ax.axhline(0, c='k', ls='--', lw=1, alpha=0.5, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([-0.5,0,0.5])
    ax.set_xlim(0, N); ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Trial #'); ax.set_ylabel('Bias')

    return ax        

def plotPsych(df, plotAvg = False, ax = None, numSessions = None, plot = 'raw'):
    
    ax = ax or plt.gca()
    
    sessIDs = df.keys()
    nBins = 15
    
    ax.set_prop_cycle(None)
    
    lw = 1
    if plotAvg:
        lw = 0.5
    
    ax.set_prop_cycle(None)
    
    for sID in sessIDs:
        
        color = next(ax._get_lines.prop_cycler)['color']

        dat = df[sID]['behavior']

        stim = dat['stimulus_frequency']
        cat = dat['stimulus_category']    
        choice = dat['choice']

        if (max(cat) == 3):    
            lb = np.mean((stim[cat==1].max(), stim[cat == 3].min()))
            ub = np.mean((stim[cat==2].min(), stim[cat == 3].max()))
        else:
            lb = stim[cat==1].max()
            ub = stim[cat==2].min()

        bounds = (lb,ub)

        binnedMean,binnedFreq, *_ = scs.binned_statistic(stim, choice, statistic = np.nanmean, bins = nBins, range=(stim.min(),stim.max()))
        binnedFreq = binnedFreq[:len(binnedFreq)-1] + (binnedFreq[1] - binnedFreq[0])/2

        if plot == 'raw':
            ax.plot(binnedFreq,binnedMean, linewidth = lw, color = color, label = 'Session' + sID)
        elif plot == 'fit':
            if (df[sID]['fit_method'] == 'bads'):    
                temp_x = np.linspace(min(stim),max(stim),50)
                params = df[sID]['fit_params']['mean']       
                f = df[sID]['function'](params, temp_x)
                ax.plot(temp_x,f, color = color, label = 'Session' + sID, zorder=4)
                ax.plot(binnedFreq,binnedMean, marker = 'o', markersize = 2, ls = (0,(1,5)), lw = 1, color = color, zorder=5)
            elif (df[sID]['fit_method'] == 'pymc'):
                yp = df[sID]['y_fit']
                ax.fill_between(yp['stimulus_freq'], yp['hdi_3%'], yp['hdi_97%'], facecolor = color, alpha=0.1, zorder=3)
                ax.plot(yp['stimulus_freq'], yp['mean'], color = color, lw=3, label = 'Session' + sID, zorder=4)
                ax.plot(binnedFreq,binnedMean, marker = 'o', markersize = 2, ls = (0,(1,5)), lw = 1, color = color, zorder=5)
               
    if plotAvg:
        temp = "behavior"
        res = [val[temp] for key, val in df.items() if temp in val]
        tdat = pd.concat(res)
        stim = tdat['stimulus_frequency']
        choice = tdat['choice']
        
        binnedMean,binnedFreq, *_ = scs.binned_statistic(stim, choice, statistic=np.nanmean, bins = nBins, range=(stim.min(),stim.max()))
        binnedFreq = binnedFreq[:len(binnedFreq)-1] + (binnedFreq[1] - binnedFreq[0])/2
        ax.plot(binnedFreq,binnedMean, color = 'k', linewidth = 2, label = 'Avg')
                  
    for b in bounds:
        ax.axvline(b, c='k', lw=0.5, alpha=0.5, zorder=0)
    ax.set_ylim([-0.05,1.05])
    ax.set_ylabel("p('High')")
    ax.set_xlabel("log(Hz)") 
    ax.axhline(0.5, c='k', ls=('--'), lw=1, alpha=0.5, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([0, 0.5, 1])

    return ax

def plotClusters(X, x_pred, nPoints, mus = None, lambdas = None, ids = None, plotAvg = True):

    un = np.unique(x_pred)

    fig, axs = plt.subplots(1,len(un), figsize = (10,2)) #(4 + 2*len(un), 2.5))

    for ia, ax in enumerate(fig.axes):
        tempt = X[x_pred == un[ia],:].T
        
        alphaT = 1
        lT = 1
        

        if plotAvg:
            alphaT = 0.2
            lTt = 4
            ax.plot(np.array(range(0,nPoints)), tempt.T[:,0:nPoints].mean(0), linewidth = lTt, color = COLORS['s_low'], zorder = 3)
            ax.plot(np.array(range(0,nPoints)), tempt.T[:,nPoints:(2*nPoints)].mean(0), linewidth = lTt, color = COLORS['s_high'], zorder = 3)

        for ii, temp in enumerate(tempt.T):

            ax.plot(np.array(range(0,nPoints)), temp[0:nPoints], linewidth = lT, color = COLORS['s_low'], alpha = alphaT, zorder = 1)
            ax.plot(np.array(range(0,nPoints)), temp[nPoints:(2*nPoints)], linewidth = lT, color = COLORS['s_high'], alpha = alphaT, zorder = 1)
        
        ax.set_ylim(0,1)
        ax.axhline(0.5, color = 'k', linestyle = '--')

    return fig, axs

def plotClustersVertThreeAx(X, X_avg, x_pred, nPoints, mus = None, lambdas = None, ids = None, plotAvg = True):

    scale1 = 0.6
    scale2 = 0.6

    un = np.unique(x_pred)

    fig, ax = plt.subplots(1, 3, figsize = (8,20))

    for ia, u in enumerate(x_pred):

        tempt = X[ia,:].T
        tempt = 1*(tempt - 0.5) + 0.5
        
        alphaT = 1
        lT = 1

        alphaT = 0.2
        lTt = 1.5

        ax[0].plot(np.array(range(0,nPoints)), ia*-scale1 + tempt.T[0:nPoints], linewidth = lTt, color = COLORS['s_low'], zorder = 3)
        ax[0].axhline(ia*-scale1 + 0.5, color = 'k', linestyle = '--', linewidth = 0.5)

        ax[0].fill_between(np.array(range(0,nPoints)), ia*-scale1 + tempt.T[0:nPoints], ia*-scale1 + 0.5, color = COLORS['s_low'], alpha = 0.5)

        ax[1].plot(np.array(range(0,nPoints)), ia*-scale1 + tempt.T[nPoints:(2*nPoints)], linewidth = lTt, color = COLORS['s_high'], zorder = 3)
        ax[1].axhline(ia*-scale1 + 0.5, color = 'k', linestyle = '--', linewidth = 0.5)

        ax[1].fill_between(np.array(range(0,nPoints)), ia*-scale1 + tempt.T[nPoints:(2*nPoints)], ia*-scale1 + 0.5, color = COLORS['s_high'], alpha = 0.5)

        tempt = X_avg[ia,:].T
        tempt = 1.2*(tempt - 0.5) + 0.5

        if x_pred[ia] == 0:
            ct = 'g'
        elif x_pred[ia] == 1:
            ct = 'm'    
        else:
            ct = [.5,.5,.5]

        ax[2].plot(np.array(range(0,nPoints)), ia*-scale2 + tempt.T[0:nPoints], linewidth = lTt, color = ct, zorder = 3)
        ax[2].axhline(ia*-scale2 + 0.5, color = 'k', linestyle = '--', linewidth = 0.5)

        ax[2].fill_between(np.array(range(0,nPoints)), ia*-scale2 + tempt.T[0:nPoints], ia*-scale2 + 0.5, color = ct, alpha = 0.5)



    return fig, ax

def plotClustersVertAvgOnly(X, x_pred, nPoints, mus = None, lambdas = None, ids = None, plotAvg = True):

    un = np.unique(x_pred)

    fig, ax = plt.subplots(1, 1, figsize = (3,20))

    ia = 0

    alphaT = 1
    lT = 1

    tempt = X[x_pred == 1,:].T

    if plotAvg:
        alphaT = 0.2
        lTt = 3
        ax.plot(np.array(range(0,nPoints)), tempt.T[:,0:nPoints].mean(0), linewidth = lTt, color = 'm', zorder = 3)

    tempt = X[x_pred == 0,:].T

    if plotAvg:
        alphaT = 0.2
        lTt = 3
        ax.plot(np.array(range(0,nPoints)), tempt.T[:,0:nPoints].mean(0), linewidth = lTt, color = 'g', zorder = 3)

        ax.set_ylim(0,1)
        ax.axhline(0.5, color = 'k', linestyle = '--')

    return fig, ax

def plotDifSizeClustersAvgOnlyOneAx(X, x_pred, nPointsVec, mus = None, lambdas = None, ids = None, plotAvg = True):

    un = np.unique(x_pred)

    fig, ax = plt.subplots(1, 1, figsize = (5,3))

    for ia, u in enumerate(x_pred):

        tempt = X[ia,:].T
        
        nPoints = int(nPointsVec[ia,0])
        print(nPoints)

        alphaT = 1
        lT = 0.5

        if plotAvg:
            alphaT = 0.2
            lTt = 4
            ax.plot(np.array(range(0,nPoints)), tempt.T[0:nPoints].mean(0), linewidth = lTt, color = COLORS['s_low'], zorder = 3)
            ax.plot(np.array(range(0,nPoints)), tempt.T[nPoints:(2*nPoints)].mean(0), linewidth = lTt, color = COLORS['s_high'], zorder = 3)
        print(tempt)

        temp = tempt
        ax.plot(np.array(range(0,nPoints)), temp[0:nPoints], linewidth = lT, color = COLORS['s_low'], alpha = alphaT, zorder = 1)
        ax.plot(np.array(range(0,nPoints)), temp[nPoints:(2*nPoints)], linewidth = lT, color = COLORS['s_high'], alpha = alphaT, zorder = 1)
        
        ax.set_ylim(0,1)
        ax.axhline(0.5, color = 'k', linestyle = '--')

    return fig, ax

def plotQModel(Q_stored, H, L, binH, binL, eOT):
    
    fig, axes = plt.subplots(3,1, figsize=(7,7))

    axes[0].plot(Q_stored[1][0:eOT], label = 'High "Value"', color = COLORS['s_high'])
    axes[0].plot(Q_stored[0][0:eOT], label = 'Low "Value"', color = COLORS['s_low'])
    axes[0].axhline(0, color = 'k', linestyle = '--')
    axes[0].set_title('Q_Values')
    axes[0].set_ylabel('Q_Value')
    axes[0].legend()
    axes[0].set_xlabel('Trial')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)

    axes[1].plot(H[0:eOT], label = 'p("H"|H)', color = COLORS['s_high'])
    axes[1].plot(1 - L[0:eOT], label = 'p("H"|L)', color = COLORS['s_low'])
    axes[1].axhline(0.25, color = 'k', linestyle = '--')
    axes[1].axhline(0.5, color = 'k', linestyle = '--')
    axes[1].axhline(0.75, color = 'k', linestyle = '--')
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Predicted Probability "High"')
    axes[1].set_xlabel('Trial')
    axes[1].legend()
    axes[1].set_title('Choice Probability')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    axes[2].plot(binH[0:eOT], label = 'p("H"|H)', color = COLORS['s_high'])
    axes[2].plot(binL[0:eOT], label = 'p("H"|L)', color = COLORS['s_low'])
    axes[2].axhline(0.25, color = 'k', linestyle = '--')
    axes[2].axhline(0.5, color = 'k', linestyle = '--')
    axes[2].axhline(0.75, color = 'k', linestyle = '--')
    axes[2].set_ylim(0,1)
    axes[2].set_ylabel('Fraction Responded "High"')
    axes[2].legend()
    axes[2].set_title('Actual Choice (Running Average, 200 Trial Bin)')
    axes[2].set_xlabel('Trial')
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)

    return fig, axes

def plot_and_wilcoxon(list1, list2, name1 = 'List 1', name2 = 'List 2', xl = None, yl = None):
    
    fig, ax = plt.subplots(1,1, figsize = (1,1))

    plt.scatter(list1, list2, 8, color = 'k')
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.axline((1,1), slope = 1, linewidth = 1, color='k', linestyle = '--', zorder = 0)
    if xl is not None:
        plt.xlim(xl)
    if yl is not None:
        plt.ylim(yl)

    # Computing the Wilcoxon signed-rank test
    stat, p = scs.wilcoxon(list1, list2)

    # Printing out the Wilcoxon signed-rank test result
    print("Wilcoxon signed-rank test:")
    print("Statistic:", stat)
    print("p-value:", p)

    return fig, ax

def plotFitSimulations(sim, reps, nPoints, smoothF, smooth2):

    sim_high = np.mean(sim['sim_high'], axis = 0)
    std_high = np.std(sim['sim_high'], axis = 0)/np.sqrt(reps)

    sim_low = np.mean(sim['sim_low'], axis = 0)
    std_low = np.std(sim['sim_low'], axis = 0)/np.sqrt(reps)

    nPs, simTrace, simTraceSigned = clf.smoothLearningTraces(sim_low, sim_high, nPoints = nPoints, smoothF = smoothF, smooth2 = smooth2)[0:3]

    nPs, simStd, x = clf.smoothLearningTraces(std_low, std_high, nPoints = nPoints, smoothF = smoothF, smooth2 = smooth2)[0:3]
    simStd[0:nPoints] = 1 - simStd[0:nPoints]

    nPs, rawTrace, rawTraceSigned = clf.smoothLearningTraces(sim['acc_low'], sim['acc_high'], nPoints = nPoints, smoothF = smoothF, smooth2 = smooth2)[0:3]

    fig, ax = plt.subplots(1,1)

    ax.plot(np.array(range(0,nPoints)), rawTrace[0:nPoints], linewidth = 2, color = getColors('s_low'), alpha = 1, zorder = 5)
    ax.plot(np.array(range(0,nPoints)), rawTrace[nPoints:(2*nPoints)], linewidth = 2, color = getColors('s_high'), alpha = 1, zorder = 5)

    ax.plot(np.array(range(0,nPoints)), simTrace[0:nPoints], linewidth = 1, linestyle = '--', color = getColors('s_low'), alpha = 1, zorder = 3)
    ax.fill_between(np.array(range(0,nPoints)), simTrace[0:nPoints] - simStd[0:nPoints], simTrace[0:nPoints] + simStd[0:nPoints], facecolor=getColors('s_low'), zorder=1, alpha=0.2)

    ax.plot(np.array(range(0,nPoints)), simTrace[nPoints:(2*nPoints)], linewidth = 1, linestyle = '--', color = getColors('s_high'), alpha = 1, zorder = 3)
    ax.fill_between(np.array(range(0,nPoints)), simTrace[nPoints:(2*nPoints)]-simStd[nPoints:(2*nPoints)], simTrace[nPoints:(2*nPoints)] + simStd[nPoints:(2*nPoints)], facecolor=getColors('s_high'), zorder=1, alpha=0.2)

    ax.set_ylim(-0.05,1.05)
    ax.axhline(0.5, color = 'k', linestyle = '--')

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.xlabel('Sessions')
    plt.ylabel('Conditional Accuracy')

    ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(1)

    fig.set_size_inches(2,1.8)

    return fig, ax