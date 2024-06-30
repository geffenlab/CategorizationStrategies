from pybads import BADS

import numpy as np
import scipy.stats as scs
import scipy.special as spc
import pandas as pd

## BADS Point Estimate Fitting

def getLL(p, stim, choice, fx):
    
    f = fx(p,stim)
    
    f[f > 0.999] = 0.999
    f[f < 0.001] = 0.001
    
    ll = 0

    lowC = f[choice == 0]
    ll -= np.sum(np.log10(1 - lowC))

    highC = f[choice == 1]
    ll -= np.sum(np.log10(highC))

    return ll

def psychFn(p, x = np.linspace(np.log2(6000),np.log2(28000),50)):

    if x.any() == None:
        x = np.linspace(np.log2(6000),np.log2(28000),50)
    if np.size(p) == 4:
        f = p[0] * p[3] + (1 - p[3]) * scs.norm.cdf(x,p[1],p[2])
    elif np.size(p) == 2:
        f = p[0] * p[1] + (1 - p[1]) * scs.norm.cdf(x, 13.8, 0.001)

    return f

defaultFitStruct = {
    'x0' : np.array([0.5, 13.5, 0.4, 0.1]),
    'lower_bounds' : np.array([0.01, 10, .008, 0]),
    'upper_bounds' : np.array([0.99, 17,  3,    1]),
    'plausible_lower_bounds' : np.array([0.2, 10.1, 0.1, 0.01]),
    'plausible_upper_bounds' : np.array([0.8, 16.9, 0.7, 0.5]),
    'fx' : psychFn,
}

trainingFitStruct = {
    'x0' : np.array([0.5, 0.1]),
    'lower_bounds' : np.array([0.01, 0]),
    'upper_bounds' : np.array([0.99, 1]),
    'plausible_lower_bounds' : np.array([0.2, 0.01]),
    'plausible_upper_bounds' : np.array([0.8, 0.5]),
    'fx' : psychFn,
}

def fitBADS(sampled, st = defaultFitStruct, nF = 1):

    options = dict(display = 'off')
    target = lambda p: getLL(p, sampled['stimulus_frequency'], sampled['choice'], fx = st['fx'])
    
    if nF == 1:
        x0 = np.zeros((1,4))
        x0[0] = st['x0']
    else:
        x0t = np.zeros((4,2))
        a = (0,1,2,3)
        for ai in a:
            x0t[ai] = [st['plausible_lower_bounds'][ai], st['plausible_upper_bounds'][ai]]

        x0 = np.array(np.meshgrid(x0t[0], x0t[1], x0t[2], x0t[3])).T.reshape(-1,4)

    fvals = np.zeros((np.size(x0,0),1))
    fit_params_t = np.zeros((np.size(x0,0),4))

    for rI, fv in enumerate(fvals):

        bads = BADS(target, x0[rI], st['lower_bounds'], st['upper_bounds'], st['plausible_lower_bounds'], st['plausible_upper_bounds'], options = options)
        optimize_result = bads.optimize()

        fit_params_t[rI] = optimize_result['x']
        fvals[rI] = optimize_result['fval']

    fvals = np.squeeze(fvals)
    fit_params_t = np.squeeze(fit_params_t)

    fit_params_t = fit_params_t[fvals.argsort()]

    fvals.sort()

    fit_params = {
        'mean' : pd.DataFrame(fit_params_t),
    }

    y_fit = {
        'fval': fvals,
    }

    function = st['fx']

    return fit_params, y_fit, function

def psychLapseReg(p, x):
    # mu, sigma, lambd, gamma
    cd = 0.5 + 0.5 * spc.erf((x-p[0])/(p[1]*np.sqrt(2)))
    yt = p[2]*p[3] + (1-p[2]) * cd
    return yt

## Ensure that training tones are not overrepresented

def rebalanceStim(bDF):
    
    probes = bDF[bDF['stimulus_category'] == 3]
    low = bDF[bDF['stimulus_category'] == 1]
    high = bDF[bDF['stimulus_category'] == 2]

    minTrials = min((len(probes),len(low),len(high)))

    if minTrials > 5:

        # Now, need to sample just that number from all ranges

        pT = probes.sample(n=minTrials, replace=False, random_state=None)
        lT = low.sample(n=minTrials, replace=False, random_state=None)
        hT = high.sample(n=minTrials, replace=False, random_state=None)

        sampledDF = pd.concat((pT,lT,hT))
    else:
        sampledDF = bDF
        
    return sampledDF

## Full Fitting Function

def getPsychFit(D, stc = 'default', nF = 1, fitMethod = 'bads'):
    
    if stc == 'training':
        st = trainingFitStruct
    else:
        st = defaultFitStruct

    saveDict = dict()
    
    stim = np.array(D['stim'])
    choice = np.array(D['y'])
    session = np.array(D['session'])
    cat = np.array(D['answer'])
    acc = np.array(D['correct'])

    for si, sID in enumerate(np.unique(session)):
        
        stimTemp = stim[session == sID]
        choiceTemp = choice[session == sID]
        catTemp = cat[session == sID]
        accTemp = acc[session == sID]

        behaviorDict = {
            'session' : sID,
            'stimulus_frequency' : stimTemp,
            'stimulus_category' : catTemp,
            'choice' : choiceTemp,
            'acc' : accTemp,
        }

        behaviorDict = pd.DataFrame(behaviorDict)
        sampled = rebalanceStim(behaviorDict)

        fit_params, y_fit, function = fitBADS(sampled, st, nF)

        sessionDict = {
            'behavior' : behaviorDict,
            'function' : function,
            'fit_params' : fit_params,
            'y_fit' : y_fit,
            'fit_method' : fitMethod,
        }       
        
        saveDict[str(sID)] = sessionDict
        
    return saveDict    


# Correlation Statistics

def calculate_pvalues_pearson(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    corr = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            res = scs.pearsonr(tmp[r], tmp[c])
            pvalues[r][c] = round(res.pvalue, 3)
            corr[r][c] = round(res.statistic, 3)
    return pvalues, corr

def calculate_pvalues_spearman(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    corr = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            res = scs.spearmanr(tmp[r], tmp[c])
            pvalues[r][c] = round(res.pvalue, 3)
            corr[r][c] = round(res.statistic, 3)
    return pvalues, corr


#### Muscimol Bootstrapping

