import numpy as np
import pandas as pd
import random
import os
import helperFns as mf

curPath = os.path.abspath(os.getcwd())

def simulateLearning(mp, stimCat = None):

    Q = mp['init_Q'].copy()

    if stimCat is None:        
        firstRand = np.array(random.choices([0,1], weights = [0.5, 0.5], k = 1))[0]
    else:
        firstRand = stimCat[0]

        mp['N'] = len(stimCat)
        stimCat = np.hstack([stimCat,(np.nan)])

    stim_stored = np.zeros((mp['N'] + 1, 2), dtype = float)
    stim_stored[0] = [-1 * (1 - firstRand), firstRand]

    Q_stored = np.zeros((2, mp['N']), dtype = float)
    choiceProb = np.zeros((mp['N']), dtype = float)

    choice = np.zeros((mp['N']), dtype = float)
    reward = np.zeros((mp['N']), dtype = float)

    ct_count = 0

    for t in range(mp['N'] - 1):

        wT = stim_stored[t] * Q + mp['bias']
        pH = 1/(1 + np.exp(-5 * np.sum(wT)))

        choiceProb[t] = pH

        choice[t] = random.choices([0,1], weights = [1 - pH, pH], k = 1)[0]

        reward[t] = 0
        if int(stim_stored[t,1]) == int(choice[t]):
            reward[t] = 1

            newRand = random.choices([0,1], weights = [0.5, 0.5], k = 1)[0]
            ct_count = 0
        else:
            reward[t] = 0
            ct_count += 1

            if ct_count > mp['max_num_cts']:
                newRand = random.choices([0,1], weights = [0.5, 0.5], k = 1)[0]
            else:
                newRand = stim_stored[t][1]

        if stimCat is not None:
            
            newRand = stimCat[t+1]

        stim_stored[t+1] = [-1 * (1 - newRand), newRand]    

        # Update Values

        delta = reward[t] - Q[int(choice[t])]

        Q[int(choice[t])] = Q[int(choice[t])] + mp['alpha'] * delta

        Q[1 - int(choice[t])] = Q[1 - int(choice[t])] - np.exp(-mp['beta']) * mp['alpha'] * delta
        
        Q_stored[:,t] = Q

    stim_stored = stim_stored[0:mp['N'],:]

    return stim_stored, choice, reward, Q_stored, choiceProb

#### fit learning

def fitLearning(guesses, mp, choice, stim_cat):

    if guesses is not None:
        if len(guesses) == 4:
            bias_guess, alpha_guess, beta_guess, q_init_guess = guesses
            mp['bias'] = bias_guess
            mp['alpha'] = alpha_guess
            mp['beta'] = beta_guess
            mp['init_Q'] = [q_init_guess, -1*q_init_guess]
        elif len(guesses) == 3:
            bias_guess, alpha_guess, q_init_guess = guesses
            mp['bias'] = bias_guess
            mp['alpha'] = alpha_guess
            mp['init_Q'] = [q_init_guess, -1*q_init_guess]
        elif len(guesses) == 2:
            bias_guess, alpha_guess = guesses
            mp['bias'] = bias_guess
            mp['alpha'] = alpha_guess

    if np.isnan(mp['bias']) or np.isnan(mp['alpha']) or np.isnan(mp['beta'] or np.isnan(mp['init_Q'])): # check inputs
        return np.inf
    else:
        mp['N'] = len(choice)

        Q = mp['init_Q'].copy()

        stim_stored = np.zeros((mp['N']  + 1, 2), dtype = float)
        firstRand = stim_cat[0]

        stim_stored[0] = [-1 * (1 - firstRand), firstRand]

        Q_stored = np.zeros((2, mp['N'] ), dtype = float)

        choiceProb = np.zeros((mp['N'] ), dtype = float)
        
        reward = 1 * np.array(choice == stim_cat)

        for t in range(mp['N']):
            
            stim_stored[t] = [-1 * (1 - stim_cat[t]), stim_cat[t]]
            
            wT = stim_stored[t] * Q + mp['bias']
            
            pH = 1/(1 + np.exp(-5 * np.sum(wT)))    
            p = (1-pH, pH)

            choiceProb[t] = p[int(choice[t])]

            # Update Values

            delta = reward[t] - Q[int(choice[t])]

            Q[int(choice[t])] = Q[int(choice[t])] + mp['alpha'] * delta
            
            if mp['beta'] < 100:
                Q[1 - int(choice[t])] = Q[1 - int(choice[t])] - np.exp(-mp['beta']) * mp['alpha'] * delta
            
            Q_stored[:,t] = Q

        negLL = -np.sum(np.log(choiceProb)) 

        return negLL

####

def generateLearningCurves(stim_stored, choice, choiceProb, learnedCutOff = 0.75):

    T = len(choice)

    trialN = np.array(range(1,T+1))

    TL = trialN[stim_stored[:,1] == 0]
    CL = choiceProb[stim_stored[:,1] == 0]
    L_inter =  1 - np.interp(trialN, TL, CL)

    TH = trialN[stim_stored[:,1] == 1]
    CH = choiceProb[stim_stored[:,1] == 1]
    H_inter = np.interp(trialN, TH, CH)

    binL = choice.copy()
    binL[stim_stored[:,1] == 1] = np.nan
    L = binL.copy()
    binL = pd.Series(binL).rolling(200,5).apply(lambda x : np.nanmean(x))

    binH = choice.copy()
    binH[stim_stored[:,1] == 0] = np.nan
    H = binH.copy()
    binH = pd.Series(binH).rolling(200,5).apply(lambda x : np.nanmean(x))

    lowest = np.min([1 - binL, binH],0)
    id = list(lowest > learnedCutOff)
    if True in id:
        eOT = id.index(True)
    else: eOT = T
    
    return H, L, binH, binL, H_inter, L_inter, eOT

def loadModelFit(num_params, top = 1, folder = 'data/RL_Data'):

    keyword = 'fits_' + str(num_params) + 'param'

    file = mf.find_files('.csv', keyword, folder)
    if len(file) > 1:
        print('Multiple files match')
        file = file[-1]

    f = pd.read_csv(file[0])

    f = f[f['Unnamed: 0'] < top]
    f['aic'] = f['nLL'] * 2 + f['D'] * 2
    f['bic'] = f['nLL'] * 2 + f['D'] * np.log(f['N'])   
    f.rename(columns = {'index':'id'}, inplace =True)
    f = f.set_index("Unnamed: 0")
    f.index.name = 'iteration'

    return f

def simulateFromFit(ID, mfN, R = 1, folder = 'data/RL_Data'):

    dataBase = os.path.abspath(os.path.join(curPath,"data/Trajectories/with_bias_learning"))

    f = loadModelFit(mfN, folder = folder)

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

    ###

    N = len(acc_high)

    model_params = {
        'N': N,
        'alpha': np.nan,
        'beta': 1000,
        'bias': np.nan,
        'init_Q': [0,0],
        'max_num_cts': 3,
    }

    model_params['bias'] = f.loc[f['id'] == ID,'bias'].iloc[0]
    model_params['alpha'] = f.loc[f['id'] == ID,'alpha'].iloc[0]

    if mfN > 2:
        qT = f.loc[f['id'] == ID,'init_Q'].iloc[0]
        model_params['init_Q'] = [qT, -1 * qT]
    if mfN > 3:
        model_params['beta'] = f.loc[f['id'] == ID,'beta'].iloc[0]

    print('Starting Learning Simulation')

    learnedCutOff = 0.75

    sim_high_mat = []
    sim_low_mat = []

    for r in range(0,R):
        
        stim_stored, choice, reward, Q_stored, choiceProb =  simulateLearning(model_params, stimCat = cat - 1)
        sim_h, sim_l, hx, lx, h_prob, l_prob, eOT = generateLearningCurves(stim_stored, choice, choiceProb, learnedCutOff = learnedCutOff) #[0:2]
        
        if len(sim_high_mat) == 0:
            sim_high_mat = sim_h
            sim_low_mat = sim_l
        else:
            sim_high_mat = np.vstack((sim_high_mat, sim_h))
            sim_low_mat = np.vstack((sim_low_mat, sim_l))

    sim_low_mat = 1 - sim_low_mat


    res = {
        'sim_high': sim_high_mat,
        'sim_low': sim_low_mat,
        'acc_high': acc_high,
        'acc_low': acc_low,
    }

    return res

def simulateFromParameters(model_params, R = 1):

    print('Starting Learning Simulation')

    learnedCutOff = 0.75

    sim_high_mat = []
    sim_low_mat = []
    stim_stored_mat = []
    choice_mat = []

    for r in range(0,R):
        
        stim_stored, choice, reward, Q_stored, choiceProb =  simulateLearning(model_params)
        sim_h, sim_l, hx, lx, h_prob, l_prob, eOT = generateLearningCurves(stim_stored, choice, choiceProb, learnedCutOff = learnedCutOff)
        
        if len(sim_high_mat) == 0:
            sim_high_mat = sim_h
            sim_low_mat = sim_l
            stim_stored_mat = stim_stored
            choice_mat = choice
        else:
            sim_high_mat = np.vstack((sim_high_mat, sim_h))
            sim_low_mat = np.vstack((sim_low_mat, sim_l))
            stim_stored_mat = np.vstack((stim_stored_mat, stim_stored))
            choice_mat = np.vstack((choice_mat, choice))

    sim_low_mat = 1 - sim_low_mat


    res = {
        'sim_high': sim_high_mat,
        'sim_low': sim_low_mat,
        'stim_stored': stim_stored_mat,
        'choice': choice_mat,
    }

    return res