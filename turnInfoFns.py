import numpy as np
import os
import pandas as pd

import brainbox2wheel as b2w

def find_files(fileend, keyword, folder):
    result = []
# Walking top-down from the root
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(fileend) and keyword in file:
                result.append(os.path.join(root, file))
    return result

##

def getTimes(taskFile, wheelFile):
    
    f = open(taskFile,'r')

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
        resp_times = np.array([timestamp[idx + 1] for idx, x in enumerate(tag) if 'RESPON' == x]) 
        start_times = np.array([timestamp[idx] for idx, x in enumerate(tag) if 'TON' == x]) 
        #end_times = np.array([timestamp[idx] for idx, x in enumerate(tag) if 'TOFF' == x])

        end_times = np.array([timestamp[idx+1] for idx, x in enumerate(tag) if 'RESPON' == x]) 

        correct_times = np.array([timestamp[idx + 1] for idx, x in enumerate(tag) if 'CORRECT' == x])

        respTags = [tag[idx + 2] for idx, x in enumerate(tag) if 'RESPON' == x and idx + 1 < len(tag)]

        correctTrials = np.array([idx for idx, x in enumerate(respTags) if 'CORRECT' == x])
        incorrectTrials = np.array([idx for idx, x in enumerate(respTags) if 'INCORRECT' == x])
        unrewardedTrials = np.array([idx for idx, x in enumerate(respTags) if 'UNREWARDED' in x])
        missedTrials = np.array([idx for idx, x in enumerate(respTags) if 'TOFF' == x])

        acc = np.zeros((len(respTags)))

        temp = correctTrials
        fill = 1
        for it in temp:
            acc[it] = fill

        temp = incorrectTrials
        fill = 0
        for it in temp:
            acc[it] = fill

        temp = missedTrials
        fill = 3
        for it in temp:
            acc[it] = fill

        temp = unrewardedTrials
        fill = 2
        for it in temp:
            acc[it] = fill

        f = open(wheelFile,'r')
        mystr = f.read()
        lines2 = [line for line in mystr.split('\n') if line.strip() != '']

        wheeltime = []
        wheelloc = []
        for x in lines2:
            wheeltime.append(x.split(' ')[1])
            wheelloc.append(x.split(' ')[2])

        lst = [eval(i) for i in wheelloc]
        wheelloc = [float(i) for i in lst] #*(360/800)

        lst = [eval(i) for i in wheeltime]
        wheeltime = [float(i)/1000000 for i in lst]

        timeDict = {
            'stim': stim_times,
            'resp': resp_times,
            'start': start_times,
            'end': end_times
        }

        wheelDict = {
            'time': wheeltime,
            'loc': wheelloc
        }
    else:
        
        timeDict = {
            'stim': [],
            'resp': [],
            'start': [],
            'end': []
        }

        wheelDict = {
            'time': [],
            'loc': []
        }
        
        acc = []
        
    return timeDict, wheelDict, acc

def getOnsetInfo(timeDict, wheelDict, threshold = [6, 2], Fs = 1000, plotsQ = False, first = None, min_gap=.1, min_dur = .05, t_thresh=.2):

    pos, t = b2w.interpolate_position(wheelDict['time'], wheelDict['loc'], freq = Fs)

    thresholds = np.array(threshold)# * (360/800)

    addEnd = 0.5
    addStart = 0.5

    if first != None:
        timeDict['end'] = timeDict['end'][0:first]

    onsetDict = dict()

    #idxs = [15, 20, 22, 27, 32]

    for ii, xi in enumerate(timeDict['end']):

        mask = np.logical_and((t < timeDict['end'][ii] + addEnd),(t > timeDict['start'][ii] - addStart))

        stats = {
            'trial': ii,
            'nturns': 0,
            'onset': [np.nan],
            'offset': [np.nan],
            'amp': [np.nan],
            'vel_t': [np.nan],
            'vel': [np.nan],
            'vel_reb_t': [np.nan],
            'vel_reb': [np.nan],
        }

        if np.sum(mask) > 10:

            onsetsT, offsetsT, peak_ampsT, peak_vel_timesT, peaksT, peak_vel_reb_timesT, peaks_rebT, axes = b2w.movements(
                t[mask], pos[mask], go = timeDict['stim'][ii], resp = timeDict['resp'][ii], pos_thresh=thresholds[0],
                pos_thresh_onset=thresholds[1], min_gap = min_gap, t_thresh = t_thresh, min_dur = min_dur, make_plots=plotsQ)

            onsetsIdx = 0

            if onsetsT.any():       
                onsetsIdx = np.logical_and(onsetsT > timeDict['stim'][ii], onsetsT < timeDict['end'][ii])

            if np.sum(onsetsIdx) > 0:
                stats = {
                    'trial': ii,
                    'nturns': np.sum(onsetsIdx),
                    'onset': onsetsT[onsetsIdx] - timeDict['stim'][ii],
                    'offset': offsetsT[onsetsIdx] - timeDict['stim'][ii],
                    'amp': peak_ampsT[onsetsIdx],
                    'vel_t': peak_vel_timesT[onsetsIdx] - timeDict['stim'][ii],
                    'vel': peaksT[onsetsIdx],
                    'vel_reb_t': peak_vel_reb_timesT[onsetsIdx] - timeDict['stim'][ii],
                    'vel_reb': peaks_rebT[onsetsIdx],                
                }

                if plotsQ:
                    print([peak_vel_timesT[onsetsIdx], peaksT[onsetsIdx]])
                    print([peak_vel_reb_timesT[onsetsIdx], peaks_rebT[onsetsIdx]])

        stats = pd.DataFrame(stats)

        if len(onsetDict) == 0:
            onsetDict = stats
        else:
            onsetDict = pd.concat((onsetDict, stats))

        if (plotsQ):
            print(ii)
            input('Press Enter to continue...')

    onsetDict.index.name = 'turn'
    onsetDict = onsetDict.reset_index()
    onsetDict['turn_inv'] = onsetDict['nturns'] - onsetDict['turn'] - 1
    onsetDict.loc[onsetDict.nturns == 0, 'turn_inv'] += 1
    onsetDict.set_index('trial')

    return onsetDict

def getVelAccelPeaks(timeDict, wheelDict, Fs = 1000):

    pos, t = b2w.interpolate_position(wheelDict['time'], wheelDict['loc'], freq = Fs)

    addEnd = 0.5
    addStart = 0.5

    accelDict = dict()
    velDict = dict()

    for ii, xi in enumerate(timeDict['end']):

        mask = np.logical_and((t < timeDict['end'][ii] + addEnd),(t > timeDict['start'][ii] - addStart))

        accelD = {
            'trial': ii,
            'accel': [np.nan],
            'accel_t': [np.nan],
            'accel_v': [np.nan],
        }

        velD = {
            'trial': ii,
            'vel': [np.nan],
            'vel_t': [np.nan],
        }

        if np.sum(mask) > 10:

            peaksAccel, peaksAccelT, peaksAccelV = b2w.accel_peaks(t[mask],pos[mask])

            accelIdx = 0
            if peaksAccel.any():
                accelIdx = np.logical_and(peaksAccelT > timeDict['stim'][ii] + 0.05, peaksAccelT < timeDict['end'][ii] + 0.05)
            if np.sum(accelIdx) > 0:
                accelD['accel'] = peaksAccel[accelIdx]
                accelD['accel_t'] = peaksAccelT[accelIdx] - timeDict['stim'][ii]
                accelD['accel_v'] = peaksAccelV[accelIdx]


            peaksVel, peaksVelT = b2w.vel_peaks(t[mask],pos[mask])

            velIdx = 0
            if peaksVel.any():
                velIdx = np.logical_and(peaksVelT > timeDict['stim'][ii] + 0.05, peaksVelT < timeDict['end'][ii] + 0.05)
            if np.sum(velIdx) > 0:
                velD['vel'] = peaksVel[velIdx]
                velD['vel_t'] = peaksVelT[velIdx] - timeDict['stim'][ii]

        accelD = pd.DataFrame(accelD)
        velD = pd.DataFrame(velD)

        if len(accelDict) == 0:
            accelDict = accelD
        else:
            accelDict = pd.concat((accelDict, accelD))

        if len(velDict) == 0:
            velDict = velD
        else:
            velDict = pd.concat((velDict, velD))

    accelDict.index.name = 'peak_accel'
    accelDict = accelDict.reset_index()
    accelDict.set_index('trial')

    velDict.index.name = 'peak_vel'
    velDict = velDict.reset_index()
    velDict.set_index('trial')

    return velDict, accelDict
