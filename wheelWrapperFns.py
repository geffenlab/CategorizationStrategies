
import os
import turnInfoFns as tfs

import pandas as pd
import numpy as np
import scipy.io as sio


def find_files(fileend, keyword, folder):
    result = []
# Walking top-down from the root
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(fileend) and keyword in file:
                result.append(os.path.join(root, file))
    return result

# Wrapper Functions

def getVelInfo(mID, keyword = "testing", last = None, txt = None):

    curPath = os.path.abspath(os.getcwd())
    base =  os.path.abspath(os.path.join(curPath,"../../projects/relevance_project/InactiveMice/"))

    fTemp = base + '/' + mID

    wheelFiles = sorted(find_files('_wheelRep.txt', keyword, fTemp))
    wheelFiles = [wf for wf in wheelFiles if ('Habituation' in wf) == False]
    
    txtEnd = '.mat'
    matFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]

    txtEnd = '.txt'
    taskFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]

    wheelFiles = [wf for iw, wf in enumerate(wheelFiles) if np.logical_and(os.path.isfile(taskFiles[iw]), os.path.isfile(matFiles[iw]))]
    
    txtEnd = '.txt'
    taskFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]
    txtEnd = '.mat'
    matFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]
    
    behaviorDict = ()
    velDict = ()
    accelDict = ()

    if len(wheelFiles) != len(taskFiles):
        print('Sort Files')
    else:
        if last != None:
            wheelFiles = wheelFiles[-last:]
            taskFiles = taskFiles[-last:]
            matFiles = matFiles[-last:]
        for ii , wheelFile in enumerate(wheelFiles):

            taskFile = taskFiles[ii]  

            timeDict, wheelDict, ac = tfs.getTimes(taskFile, wheelFile)
            
            matFile = matFiles[ii]

            mat_contents = sio.loadmat(matFile)
            
            stimf = mat_contents['tt'][0:np.size(ac),4]
            cat = mat_contents['tt'][0:np.size(ac),0]

            wd = mat_contents['wheelDir'][0,0:np.size(ac)] - 1
            
            if len(timeDict["stim"]) > 5:
                if txt != None:
                    print('Starting Session ' + wheelFile)

                velDictTemp, accelDictTemp = tfs.getVelAccelPeaks(timeDict, wheelDict)
                
                velDictTemp['session'] = ii
                accelDictTemp['session'] = ii

                velDictTemp['filename'] = taskFile.split('_')[-2]
                accelDictTemp['filename'] = taskFile.split('_')[-2]
                
                if len(velDict) == 0:
                    velDict = velDictTemp
                else:
                    velDict = pd.concat((velDict, velDictTemp))
                    
                if len(accelDict) == 0:
                    accelDict = accelDictTemp
                else:
                    accelDict = pd.concat((accelDict, accelDictTemp))

                tempDic = {
                    'session': ii,
                    'accuracy': ac,
                    'stimulus_frequency': stimf,
                    'wheelDir': wd,
                    'cat': cat,
                }
                tempDic = pd.DataFrame(tempDic)
                
                tempDic.index.name = 'trial'
                tempDic = tempDic.reset_index()

                if len(behaviorDict) == 0:
                    behaviorDict = tempDic
                else:
                    behaviorDict = pd.concat((behaviorDict, tempDic))

    return behaviorDict, velDict, accelDict

def getMovementInfo(mID, thresh = (6, 2), keyword = "testing", last = None, txt = None, plotsQ = False, min_gap = .01, min_dur = .05, t_thresh = .1):

    curPath = os.path.abspath(os.getcwd())
    base =  os.path.abspath(os.path.join(curPath,"../../projects/relevance_project/InactiveMice/"))
    
    fTemp = base + mID
    #keyword = "testing"

    wheelFiles = sorted(find_files('_wheelRep.txt', keyword, fTemp))

    txtEnd = '.mat'
    matFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]

    txtEnd = '.txt'
    taskFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]

    peakDict = ()
    onsetDict = ()
    behaviorDict = ()
    accelDict = ()
    #sio.loadmat(file_name, mdict=None, appendmat=True, **kwargs)


    wheelFiles = [wf for iw, wf in enumerate(wheelFiles) if np.logical_and(os.path.isfile(taskFiles[iw]), os.path.isfile(matFiles[iw]))]
    
    txtEnd = '.txt'
    taskFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]
    txtEnd = '.mat'
    matFiles = [str(wheelFile[:-13]) + str(txtEnd) for wheelFile in (wheelFiles)]

    if len(wheelFiles) != len(taskFiles):
        print('Sort Files')
    else:
        if last != None:
            wheelFiles = wheelFiles[-last:]
            taskFiles = taskFiles[-last:]
            matFiles = matFiles[-last:]
        for ii , wheelFile in enumerate(wheelFiles):

            breakpoint()
            #print(str(ii+1) + "/" + str(len(wheelFiles)))
            taskFile = taskFiles[ii]  

            timeDict, wheelDict, ac = tfs.getTimes(taskFile, wheelFile)
            
            matFile = matFiles[ii]
            
            mat_contents = sio.loadmat(matFile)
            
            stimf = mat_contents['tt'][0:np.size(ac),4]
            cat = mat_contents['tt'][0:np.size(ac),0]

            wd = mat_contents['wheelDir'][0,0:np.size(ac)] - 1
            
            if len(timeDict["stim"]) > 5:
                if txt != None:
                    print('Starting Session ' + wheelFile)

                onsetDictTemp, accelDictTemp = tfs.getOnsetInfo(timeDict, wheelDict, threshold = thresh, 
                min_gap=min_gap, min_dur = min_dur, t_thresh=t_thresh, plotsQ = plotsQ, accelOnly = False)
                
                onsetDictTemp['session'] = ii
                accelDictTemp['session'] = ii

                onsetDictTemp['filename'] = taskFile.split('_')[-2]

                if len(onsetDict) == 0:
                    onsetDict = onsetDictTemp
                else:
                    onsetDict = pd.concat((onsetDict, onsetDictTemp))
                    
                if len(accelDict) == 0:
                    accelDict = accelDictTemp
                else:
                    accelDict = pd.concat((accelDict, accelDictTemp))

                tempDic = {
                    'session': ii,
                    'accuracy': ac,
                    'stimulus_frequency': stimf,
                    'wheelDir': wd,
                    'cat': cat,
                }
                tempDic = pd.DataFrame(tempDic)
                
                tempDic.index.name = 'trial'
                tempDic = tempDic.reset_index()

                if len(behaviorDict) == 0:
                    behaviorDict = tempDic
                else:
                    behaviorDict = pd.concat((behaviorDict, tempDic))

    return onsetDict, behaviorDict, accelDict         



def velocityThresholding(df, thresh, t_win = (0.1, 0.8)):

    sessions = np.unique(df['session'])

    choice = []
    cat = []
    wd = []
    rt = []
    session = []
    trial = []
    f_vel = []
    freq = []

    for iis, si in enumerate(sessions):

        temp1 = df[df['session'] == si]

        trials = np.unique(temp1['trial'])

        for ii, ti in enumerate(trials):

            temp = temp1[temp1['trial'] == ti]

            velT = temp['vel'].astype(str).astype(float)
            velT_abs = np.abs(velT.copy())
            vel_t = temp['vel_t'].astype(str).astype(float)

            temp2 = velT_abs[np.logical_and(vel_t > t_win[0], vel_t < t_win[1])]

            if len(temp2) > 0:
                if np.max(temp2) < thresh:
                    choice.append(np.nan)
                    rt.append(np.nan)
                    f_vel.append(np.nan)

                else: 
                    id = np.argmax(np.logical_and(velT_abs > thresh, np.logical_and(vel_t > t_win[0], vel_t < t_win[1])))
                    choice.append(np.sign(velT.iloc[id])) 
                    f_vel.append(velT.iloc[id])
                    rt.append(vel_t.iloc[id])
            else:
                choice.append(np.nan)
                rt.append(np.nan)
                f_vel.append(np.nan)          

            session.append(temp['session'].iloc[0])
            trial.append(temp['trial'].iloc[0])
            cat.append(temp['cat'].iloc[0])
            wd.append(temp['wheelDir'].iloc[0])
            freq.append(temp['stimulus_frequency'].iloc[0])

    dfF = {
        'session': session,
        'trial': trial,
        'choice': choice,
        'rt': rt,
        'f_vel': f_vel,
        'freq': freq,
    }

    dfF = pd.DataFrame(dfF)
    return dfF
