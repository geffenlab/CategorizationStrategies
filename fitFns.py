## Set of functions for fitting psychometric curves

from pybads import BADS

import numpy as np
import scipy.stats as scs
import scipy.special as spc
import pandas as pd

def psychFn(p, x = np.linspace(np.log2(6000),np.log2(28000),50)):

    '''
    transforms frequency values into p("High") values based on parameters p

    Args: 
    p: parameters to function fx, structured as [gamma mu sigma lambda]
    x: stimulus frequencies: default is 50 values spanning the frequency range for visualizing the full psychometric
        function, but this function can also be used to return the p("H") for presented stimuli for fitting purposes

    Returns:
    f: p("H") corresponding to each frequency value in x
    '''

    if x.any() == None:
        x = np.linspace(np.log2(6000),np.log2(28000),50)
    if np.size(p) == 4:
        f = p[0] * p[3] + (1 - p[3]) * scs.norm.cdf(x,p[1],p[2])
    elif np.size(p) == 2:
        f = p[0] * p[1] + (1 - p[1]) * scs.norm.cdf(x, 13.8, 0.001)

    return f

# Define fit structures for fitting testing or training data with psychometric function

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

def getLL(p, stim, choice, fx):

    '''
    calculates negative log-likelihood from psychometric function parameters and behavioral data

    Args: 
    p: parameters to function fx
    stim: stimulus input to function fx
    choice: actual behavioral choice to stimulus stim
    fx: function handle for psychometric function

    Returns:
    ll: negative log-likelihood to potentially minimize
    '''

    f = fx(p,stim)
    
    f[f > 0.999] = 0.999
    f[f < 0.001] = 0.001
    
    ll = 0

    lowC = f[choice == 0]
    ll -= np.sum(np.log10(1 - lowC))

    highC = f[choice == 1]
    ll -= np.sum(np.log10(highC))

    return ll




def fitBADS(behavior, st = defaultFitStruct, nF = 1):

    '''
    fit psychometric function to data using PyBADS

    Args: 
    behavior: pandas dataframe with 'stimulus frequency' and 'choice' columns
    st: structure for fit, with fit initialization and boundary parameters
    nf: number of runs from different starting points: default is 1, currently structured so anything other than 1 is, well, 2 (so 2^4 = 16 runs)
        - this currently is set up to work with the defaultFitStruct, not the trainingFitStruct, but can be easily changed by editing line 108-9

    Returns:
    fit_params: fit parameters for each run (sorted by negative log-likelihood)
    y_fit: associated negative log-likelihoods
    function: function used, so that the p("H") values can be recreated
    '''

    options = dict(display = 'off')
    target = lambda p: getLL(p, behavior['stimulus_frequency'], behavior['choice'], fx = st['fx'])
    
    if nF == 1:
        x0 = np.zeros((1,4))
        x0[0] = st['x0']
    else:
        x0t = np.zeros((4,2)) # <- edit here to (2,2) if running multiple starting points on training data fit structure (so not fitting threshold or sigma)
        a = (0,1,2,3) # <- edit here to (0,1) if running multiple starting points on training data fit structure (so not fitting threshold or sigma)
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

## Ensure that training tones are not overrepresented

def rebalanceStim(bDF):

    '''
    ensures that training tones are not overrepresented by sampling the minimum number of presented stimuli from all 3 categories

    Args: 
    bDF: behavioral dataframe with 'stimulus_category' column

    Returns:
    sampledDF: behavioral dataframe, now balanced so that there's the same number of trials for low/probe/high
    '''   

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

    '''
    Fitting wrapper to take behavioral data, fit it, and output fit results

    Args: 
    D: Behavioral dictionary as outputted by, for example, 'analysis_GenerateTrajectories'. Contains curated behavioral data and session indices
    stc: structure option: if you only want to fit training data without probes, input "training" and the psychometric threshold and sigma won't
        be fit, just the guess rate and lapse rate. default is to fit all 4 parameters of the psychometric function
    nF: number of runs from different starting points: default is 1, currently structured so anything other than 1 is, well, 2 (so 2^4 = 16 runs)
        - this currently is set up to work with the defaultFitStruct, not the trainingFitStruct, but can be easily changed by editing line 108-9
    fitMethod: Only option right now is 'bads', can be modified to switch from point estimates
        to distribution fitting with pymc if preferred

    Returns:
    saveDict: essentially returns information from dictionary D, but separated based on session and with fit results/nLL included. Individual
        session information can be accessed through saveDict['1'] for session 1, etc.
    '''    

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

def calculate_pvalues_spearman(df):

    '''
    Calculate spearman's rho correlation coefficient and associated p-value

    Args: 
    df: dataframe with two columns to be correlated against each other

    Returns:
    pvalues: p-values of correlation
    corr: Spearman's rho
    '''

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