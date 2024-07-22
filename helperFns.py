## Set of functions for loading, curating and transforming data

import numpy as np
import scipy.io as spio
import os
import pandas as pd
import pickle
import scipy.stats as scs

def find_files(fileend, keyword, folder):

    '''
    Locates desired files

    Args:
    fileend: type of file required (for example, '.csv')
    keyword: string to look for in file name
    folder: folder to start in (but will walk through subfolders)

    Returns:
    result: list of files that match query
    '''

    result = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(fileend) and keyword in file:
                result.append(os.path.join(root, file))
    return result

def _todict(matobj):
    '''
    Thanks to 'mergen', from StackOverflow: a recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict):
    '''
    Thanks to 'mergen', from StackOverflow: changes mat-objects to nested dicts
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def loadmat(filename):
    '''
    Thanks to 'mergen', from StackOverflow: cures all entries from spio.loadmat that aren't changed
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def loadSavedFits(ID, dataBase, ending = '_trainingDataBias'):

    '''
    Locates and loads dictionaries created from, for example, analysis_GenerateTrajectories

    Args:
    ID: mouse ID to check
    dataBase: folder to start with
    ending: keyword to specify what type of trajectories we're extracting

    Returns:
    loaded_dict: dictionary with (likely) weights, choice behavior, and potentially psychometric fit results
    '''
    
    file = find_files('.pkl', ID + ending, dataBase)[0]
    with open(file, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def getD(ID, keyword = "training", cutoff = 50, transformation_weight = 2, sessionCutoff = None, 
         accCutoff = 0, removeProbes = True, nrCheck = False, txt = None, untilTesting = True, spec_select = False):

    '''
    Locates and loads raw behavioral matlab files, and transforms them to the format needed for PsyTrack. 

    Args:
    ID: mouse ID to check
    keyword: string to look for in behavior files: default is "training", but "testing" is another option
    cutoff: minimum number of non-probe trials
    transformation_weight: tau for tau transformation of weights
    sessionCutoff: minimum number of trials in session (default is None)
    accCutoff: minimum accuracy on non-probe trials across session
    removeProbes: whether to consider probe trials: not an issue on training data, but on testing data, if interested in doing direct comparisons to training data, can be useful to remove
    nrCheck: whether to also extract information about the no-response probabilities, which are removed in this curation process: if True, also returns nrDict
    txt: whether to output updates to the console
    untilTesting: whether to use all data up until testing sessions or further curate the data to have trajectories end at similar accuracy points
    spec_select: if False, excludes sessions that, at time of behavior, were aborted or marked as unusable due to external factors

    Returns:
    D: dictionary with curated choice behavior in a format to be fed into psytrack, or fit with RL model, or clustered.
    '''

    curPath = os.path.abspath(os.getcwd())
    base =  os.path.abspath(os.path.join(curPath,"data/MouseData/"))

    fTemp = base + '/' + ID

    fTemp = os.path.normpath(fTemp)
    matFiles = sorted(find_files(".mat", keyword, fTemp))
    matFiles = [mf for mf in matFiles if ('Hab' in mf) == False]

# Ensure that known anomalies are accounted for

    matFiles = curateSessions(matFiles,ID,keyword, untilTesting)

    txtEnd = '.txt'
    taskFiles = [str(matFile[:-4]) + str(txtEnd) for matFile in (matFiles)]

    matFiles = [mf for im, mf in enumerate(matFiles) if os.path.isfile(taskFiles[im])]

    txtEnd = '.txt'
    taskFiles = [str(matFile[:-4]) + str(txtEnd) for matFile in (matFiles)]

# Initialize vectors
    
    s_h = []
    s_l = []
    s_avg = []
    c = []
    y = []
    correct = []
    rt = []
    answer = []
    stims = []
    dayLength = []
    sessDate = []
    session = []

    sessionN = 1
    nrDict = dict()

# Loop over session files

    for ii , matFile in enumerate(matFiles):
        
        #print(matFile)

        taskFile = taskFiles[ii]
        
        tmp = taskFile.split('_')

        sessDateT = tmp[-2]
        
        mat_contents = spio.loadmat(matFile)
        good = True

# Some sessions were marked as unusable at the time of behavior, or soon afterwards (for example, if there was an
# error on the side of the experimenter, or if a mouse performed poorly and health issues were discovered soon 
# afterwards). 

        if spec_select == False:

            if (sessionCutoff != None):
                if (np.size(dayLength) > (sessionCutoff - 1)):
                    good = False
            
            if ('excludeTag' in mat_contents):
                if (mat_contents['excludeTag'][0] == 'TRUE'):
                    good = False
                    
            if ('testingBiased' in matFile):
                good = False

            if ('message' in mat_contents):
                if (np.size(mat_contents['message']) > 0):
                    if ('MUSC' in mat_contents['message'][0].upper()):
                        good = False

        temp = np.array(mat_contents['resp'].copy()).squeeze()
        temp = temp[temp < 2]

        if len(temp) < cutoff:
            good = False

        if good:
            ac = np.array(mat_contents['resp'].copy()).squeeze()
            acT = ac[np.logical_and(ac > -1, ac < 2)]

            if np.mean(acT) < accCutoff:
                good = False

        if good:        

            if ('trialEndPoint' in mat_contents):
                endPt = mat_contents['trialEndPoint'].squeeze().astype(int) - 1
            else:
                endPt = np.size(ac) - 1

            if ('trialStartPoint' in mat_contents):
                startPt = mat_contents['trialStartPoint'].squeeze().astype(int) - 1
            else:
                startPt = 0         

            ac = ac[startPt:endPt]            

            stimulus_category = mat_contents['tt'][startPt:endPt,0].copy()

    # For the oldest stimulus files, the "category" is indexed from 1-7, where 1-2 are "Low", 6-7 are "High" and
    # 3-5 are "Probe". For the majority of the files, the "category" is simply indexed from 1-3, where 1 is "Low",
    # 3 is "High" and 2 is "Probe". Here, we transform to "High = 1" and "Low = 0".
            
            if (max(stimulus_category) == 3):
                stimulus_category[stimulus_category == 1] = 0
                stimulus_category[stimulus_category == 3] = 1
                stimulus_category[stimulus_category == 2] = 99
            elif (max(stimulus_category) == 7):
                stimulus_category[stimulus_category < 3] = 0
                stimulus_category[stimulus_category > 5] = 1
                stimulus_category[stimulus_category > 1] = 99  
            else:
                print('Error ' + matFile)    

    # For the oldest session files, recreate the "choice" vector by using the recorded trial category and accuracy.
    # Otherwise, use the "wheelDir" vector.


            if ('wheelDir' in mat_contents):
                wDir = mat_contents['wheelDir'][0,startPt:endPt].copy()
                choiceT = getResp(ac, stimulus_category, mat_contents, respDir = wDir)
            elif ('lickDir' in mat_contents):
                lDir = mat_contents['lickDir'][0,startPt:endPt].copy()
                choiceT = getResp(ac, stimulus_category, mat_contents, respDir = lDir)
            else:
                choiceT = getResp(ac, stimulus_category, mat_contents)
            
            rtt = getRT(taskFile)

            if len(rtt) == 0:
                rtt = np.empty(np.size(choiceT))
                rtt[:] = np.nan

            rtt = rtt[startPt:endPt]
            
            t = np.array(range(startPt,endPt))

            if (len(choiceT) > len(rtt)):
                choiceT = choiceT[0:len(rtt)]
                ac = ac[0:len(rtt)]
                stimulus_category = stimulus_category[0:len(rtt)]
                t = t[0:len(rtt)]
            elif (len(choiceT) < len(rtt)):
                rtt = rtt[0:len(choiceT)]
                ac = ac[0:len(choiceT)]
                stimulus_category = stimulus_category[0:len(choiceT)]
                t = t[0:len(choiceT)]

            goodtrial = np.array(~np.isnan(choiceT))

            acc = ac[goodtrial].copy()
            rtt = rtt[goodtrial].copy()
            t = t[goodtrial].copy()

    # We also don't want to include sessions that have too few trials to reliably "fit".

            if (np.size(acc) < cutoff):
                good = False

            if good:

    # We design a vector that simply represents whether a given trial was preceded by a valid trial.

                prior = ((t[1:] - t[:-1]) == 1).astype(int)
                prior = np.hstack(([0], prior))

                choice = choiceT[goodtrial].copy()
                stimulus_category = stimulus_category[goodtrial]       

    # As a check to ensure that our stimulus category and choice vectors are properly coded, we calculate the accuracy
    # on trials where the stimulus category and choice match. If the accuracy is 100%, we're good to go.

                tmp = np.array((stimulus_category[choice < 2]  == choice[choice < 2]))

                switch = np.mean(acc[tmp])

                if (switch < 0.5):
                    print('Coding Error ' + matFile)
                    choice = 1 - choice
                elif (sum(tmp) == 0):
                    print('Empty Error ' + matFile)
                elif (np.isnan(switch)):
                    print('Error ' + matFile)

    # The specific way that the stimulus frequency is "saved" has evolved over time. For the oldest sessions, the stimuli
    # must be accessed through "indices" saved in the .mat file that allow for the location of the actual frequency in a 
    # different stimulus file.

                if (np.size(mat_contents['tt'][0]) < 5):
                    stim = getStimFromIdx(mat_contents)
                    stim = stim[startPt:endPt]
                elif (np.size(mat_contents['tt'][0]) == 5):
                    stim = np.log2(mat_contents['tt'][startPt:endPt,4]).copy()
                elif (np.size(mat_contents['tt'][0]) == 6):
                    stim = np.log2(mat_contents['tt'][startPt:endPt,3]).copy()
                else:
                    stim = np.log2(mat_contents['tt'][startPt:endPt,4]).copy()


                if np.max(stim) > 10000:
                    stim = np.log2(stim)

    # Here, I'm interested in just capturing the "no response rate" for varying frequencies, to determine whether there's a 
    # psychometric shape that would indicate the "no response rate" is linked to a turn direction.

                if nrCheck:
                    uniqueStims = np.unique(stim)
                    nrNum = [np.sum(np.isnan(choiceT[stim == s_i])) for s_i in uniqueStims]
                    N = [len(choiceT[stim == s_i]) for s_i in uniqueStims]
                    nrProbs = [np.mean(np.isnan(choiceT[stim == s_i])) for s_i in uniqueStims]
                    temp = {'stims' : uniqueStims,
                            'nrNum' : nrNum,
                            'N' : N,
                            'nrProbs' : nrProbs,
                            } 
                    nrDict[str(sessionN)] = pd.DataFrame(temp)

    # Because we see that mice often learn one side association before the other, we will treat the "evidence towards high"
    # and "evidence towards low" as separate predictors, that can evolve independently.
                
                stim = stim[0:len(goodtrial)]
                
                stimH = stim.copy()
                stimH = (stimH - min(stimH))
                stimH = stimH / (max(stimH)-min(stimH))
                
                stimH = stimH * 2
                stimH = stimH - 1

                stimH = stimH[goodtrial]

                stimL = -1 * stimH.copy()

                stimH[stimulus_category == 0] = 0
                stimH = np.tanh(transformation_weight*stimH)/np.tanh(transformation_weight)

                stimL[stimulus_category == 1] = 0        
                stimL = np.tanh(transformation_weight*stimL)/np.tanh(transformation_weight)

    # Previous "average tone value", we'll just say this is the normalized tone value over the full range of stimuli that
    # session.

                stimAvg = np.array(stim.copy())
                stimAvg = stimAvg[goodtrial]

                stimAvg = stimAvg[:-1]
                stimAvg = (stimAvg - np.mean(stimAvg))/np.std(stimAvg)
                stimAvg = np.hstack(([0], stimAvg))
                stimAvg = stimAvg * prior

    # Previous choice.

                ct = (choice.copy()[:-1] * 2 - 1).astype(int)
                ct = np.hstack(([0], ct))
                ct = ct * prior

                stim = np.array(stim)

    # Win-Stay Effect: previous choice * reward (so, 1 if previous choice was left and rewarded, -1 if )

                temp = (choice.copy() * 2 - 1) * acc.copy()
                ws = (temp[:-1]).astype(int)
                ws = np.hstack(([0], ws))

    # Lose-Switch Effect: previous choice * (1-reward) (so, 1 if previous choice was left and an error, -1 if )

                temp = (choice.copy() * 2 - 1) * (1 - acc.copy())
                ls = (temp[:-1]).astype(int)
                ls = np.hstack(([0], ls))

    # Appending current session data to full arrays.

                stim = np.array(stim[goodtrial])

                if removeProbes:
                    trainingIdx = np.array(stimulus_category < 2)

                    stim = stim[trainingIdx]
                    stimH = stimH[trainingIdx]
                    stimL = stimL[trainingIdx]
                    stimAvg = stimAvg[trainingIdx]
                    ct = ct[trainingIdx]
                    ws = ws[trainingIdx]
                    ls = ls[trainingIdx]
                    choice = choice[trainingIdx]
                    acc = acc[trainingIdx]
                    rtt = rtt[trainingIdx]
                    stimulus_category = stimulus_category[trainingIdx]

                stims.extend(stim)
        
                s_h.extend(np.array(stimH))
                s_l.extend(np.array(stimL))

                s_avg.extend(np.array(stimAvg)) # Previous Stim
                c.extend(np.array(ct)) # Previous Choice

                y.extend(np.array(choice))
                correct.extend(np.array(acc))
                rt.extend(np.array(rtt))

                stimulus_category += 1
                stimulus_category[stimulus_category > 2] = 3
                
                answer.extend(np.array(stimulus_category))
                
                session.extend(np.array([sessionN]*np.size(acc)))

                sessDate.append(sessDateT)

                dayLength.append(np.size(ct))
                
                if (txt != None):
                    print('Finished ' + matFile)
                sessionN += 1

    inputs = dict(s_high = np.array(s_h)[:, None],
                    s_low = np.array(s_l)[:, None],
                    s_avg = np.array(s_avg)[:, None],
                    c = np.array(c)[:, None])

    D = dict(
        subject = ID,
        stim = stims,
        inputs = inputs,
        s_high = np.array(s_h),
        s_low = np.array(s_l),
        correct = np.array(correct),
        answer = np.array(answer),
        rt = np.array(rt),
        y = np.array(y),
        session = np.array(session),
        sessDate = np.array(sessDate),
        dayLength = np.array(dayLength),
    )

    if nrCheck:
        return D, nrDict
    else:
        return D

def curateSessions(matFiles,ID,keyword, untilTesting):

    '''
    Curates sessions based on desired choices

    Args:
    matFiles: vector of files to consider
    ID: mouse ID
    keyword: only want to curate training data, don't want to mess with the testing data
    untilTesting: if untilTesting, include sessions up until testing probes introduced. Otherwise, include up until accuracy threshold reached.
        This comes up when, for example, a mouse reaches the threshold on a Friday, and we want to make sure the behavior is stable on Monday so
        we run an additional training session.

    Returns:
    curated matFiles vector
    '''  

    if keyword == "training":
        if untilTesting == False:
            match ID:
                case 'GS027':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210811)]
                case 'GS028': 
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210524)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210716)]            
                case 'GS029':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210728)] 
                case 'GS030':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210812)]
                case 'GS037':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210702)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210817)]
                case 'GS040':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210702)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 210810)]
                case 'JC025': # For JC025, remove the sessions post-5/08/21, as the mouse was re-trained on a different set.
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) < 210508)]
                case 'JC028': # For JC028, remove the sessions post-5/08/21, as the mouse was re-trained on a different set.
                    matFiles = [matFile for matFile in matFiles if ('210416' not in matFile)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) < 210508)]            
                case 'JC029':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210602)] 
                case 'JC044':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211128)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 211229)]
                case 'JC047':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211109)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 211201)]
                    matFiles = [matFile for matFile in matFiles if ('211128' not in matFile)]
                case 'JC048':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211109)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 211214)]
                case 'JC052':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211130)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 211228)]
                case 'JC057':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220125)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 220218)]
                case 'JC059':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220126)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 220225)]
                case 'JC061':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220228)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 220418)]
                case 'JC062':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220228)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 220406)]
                case 'JC067':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220504)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) <= 220530)]
        else:
            match ID:
                case 'GS027':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                case 'GS028':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210524)]        
                case 'GS029':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                case 'GS030':
                    matFiles = [matFile for matFile in matFiles if ('Lowered' not in matFile)]   
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) > 210630)]
                case 'GS037':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210702)]
                case 'GS040':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210702)]
                case 'JC025': # For JC025, remove the sessions post-5/08/21, as the mouse was re-trained on a different set.
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) < 210508)]
                case 'JC028': # For JC028, remove the sessions post-5/08/21, as the mouse was re-trained on a different set.
                    matFiles = [matFile for matFile in matFiles if ('210416' not in matFile)]
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) < 210508)]            
                case 'JC029':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 210602)] 
                case 'JC044':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211128)]
                case 'JC047':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211109)]
                    matFiles = [matFile for matFile in matFiles if ('211128' not in matFile)]
                case 'JC048':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211109)]
                case 'JC052':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 211129)]
                case 'JC057':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220125)]
                case 'JC059':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220126)]
                case 'JC061':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220228)]
                case 'JC062':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220228)]
                case 'JC067':
                    matFiles = [matFile for matFile in matFiles if (int(matFile.split('_')[-2]) >= 220504)]
    return matFiles

def getStimFromIdx(mat_contents):

    '''
    For some older mice, stimulus index is saved as a separate file, so we need to access it to determine the stimuli used

    Args:
    mat_contents: matlab file of raw behavioral data and session information

    Returns:
    stim: vector of stimuli presented during session
    '''

    curPath = os.path.abspath(os.getcwd())
    stim_base = os.path.abspath(os.path.join(curPath,"data/Stimuli/"))

    stimFile = stim_base + '/' + mat_contents['params']['stimName'][0][0][0]
    stimFile = os.path.normpath(stimFile)

    stim_contents = spio.loadmat(stimFile)
    stim_contents = stim_contents['stims']['t'][0]

    stim_idx = mat_contents['tt'][:,0:2]
    stim = [stim_contents[sI[0]-1][sI[1]-1][0] for i, sI in enumerate(stim_idx)]

    return stim

def getResp(acc, stimulus_category, mat_contents, respDir = ()):    

    '''
    Get choice response of mice from behavioral information

    Args:
    acc: vector of whether mouse got trial correct
    stimulus_category: vector of presented category
    mat_contents: matlab file of raw behavioral data to check reward contingency (High -> CW or CCW)
    respDir: for some older mice, choice was coded as response direction

    Returns:
    choice: vector of mouse response choices 
    '''

    choice = np.array(stimulus_category.copy())
    choice[acc == 0] = 1 - choice[acc == 0]
    choice = np.where(acc == 2, np.nan, choice)

    if  len(respDir) > 0:

        respDir -= 1
        
        if (mat_contents['params']['rewardContingency'][0][0][0][0] == 2):
            respDir = 1 - respDir

        choice[choice > 10] = respDir[choice > 10]

    return choice

def getRT(taskFile):

    '''
    Get trial response times from task file.

    Args:
    taskFile: csv file from behavior session that includes time stamps for stimulus onsets and choices

    Returns:
    rt: response times for each trial for the session
    '''

    try:

        with open(taskFile, 'r') as f:
            mystr = f.read()

        lines = [line for line in mystr.split('\n') if line.strip() != '']

        for ii, tx in enumerate(lines):
            if tx.split(' ')[0] == '0001':
                startPt = ii
                break
        else:
            startPt = None
        
        lines = lines[startPt:]

        trialN = []
        timestampT = []
        tag = []

        for x in lines:
            tagT = x.split(' ')[2]

            if np.logical_and('HOLDSTART' not in tagT, 'WHEELTURN' not in tagT):
                trialN.append(x.split(' ')[0])
                timestampT.append(x.split(' ')[1])
                tag.append(tagT)

        lst = [eval(i) for i in timestampT]
        timestamp = [float(i)/1000000 for i in lst]

        if np.max(timestamp) > 4200:
            timestamp = np.array([ts if ts > 1000 else ts + 4294.967295 for ts in timestamp])

        respOn_idx = np.array([[idx] for idx, x in enumerate(tag) if 'RESPON' == x]) 
        
        if len(respOn_idx) > 50:
        
            lastRespOn = respOn_idx[-1]

            if lastRespOn + 3 > len(tag):
                trial_idx = np.array([[idx] for idx, x in enumerate(tag) if 'TRIAL' in x]) 
                lastTrial = trial_idx[-1]
                tag = tag[0:(lastTrial[0])]

            stim_times = np.array([timestamp[idx] for idx, x in enumerate(tag) if 'STIMON' == x])
            response_times = np.array([timestamp[idx + 1] for idx, x in enumerate(tag) if 'RESPON' == x and idx + 1 < len(tag)])
            stim_times = stim_times[0:len(response_times)]
            rt = response_times - stim_times

    except UnicodeDecodeError:
        print('Unicode Error')
        rt = ()
    except:
        print('Something else went wrong')
        rt = ()

    return rt

def muscimolStruct():

    '''
    Generate structure with information about muscimol and control session dates

    Args: 
    None, just called when needed to access relevant dates

    Returns:
    injectionDates: nested dictionary of each mouse's dates for control and inactiviation sessions
    '''

    injectionDates = {}

    id = 'JC044'
    temp = {
        'pbs_dates' : ['220120','220125','220131','220202'],
        'mus_dates' :['220119','220126','220201','220204']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC047'
    temp = {
        'pbs_dates' : ['220121','220128','220131'],
        'mus_dates' :['220124','220126','220201']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC052'
    temp = {
        'pbs_dates' : ['220120','220125','220128','220202'],
        'mus_dates' :['220119', '220121', '220126','220201']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC057'
    temp = {
        'pbs_dates' : ['220222','220228','220304','220308'],
        'mus_dates' :['220223','220301','220305','220309']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC059'
    temp = {
        'pbs_dates' : ['220314','220322','220325'],
        'mus_dates' :['220315','220323','220326']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC061'
    temp = {
        'pbs_dates' : ['220421','220425','220428'],
        'mus_dates' :['220422','220426','220429']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    id = 'JC062'
    temp = {
        'pbs_dates' : ['220421','220425','220429'],
        'mus_dates' :['220422','220427','220503']
        }
    temp = pd.DataFrame(temp)

    injectionDates[id] = temp

    return injectionDates

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
            pvalues.loc[c,r] = round(res.pvalue, 3)
            corr.loc[c,r] = round(res.statistic, 3)

    return pvalues, corr