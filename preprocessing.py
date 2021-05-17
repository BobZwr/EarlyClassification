import numpy as np
import os
import wfdb
from collections import Counter
import pickle
import random
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d

from scipy.signal import butter, lfilter
from tqdm import tqdm
from scipy.interpolate import interp1d

flatten = lambda t: [item for sublist in t for item in sublist]
fs_out = 200

def get_time_to_sepsis(label):
    Label = []
    end = -1
    for person in label:
        new_person = []
        if sum(person) == 0:
            for i in range(len(person)):
                new_person.append([0, (len(person) - i) / fs_out])
            Label.append(new_person)
            continue
        current_va = 0
        for i in reversed(range(len(person))):
            if person[i] == 1:
                current_va = i
            else:
                if current_va == 0:
                    continue
                new_person.append([1, (current_va - i) / fs_out])
        new_person.reverse()
        Label.append(new_person)
    return Label


def read_data( filename ):
    '''
     *** Description ***

     Read the data in filename
     Return the dictionary Data & Label grouped by the pid
    '''

    Data = dict()
    Label = dict()
    data = np.load('data/' + filename + '_data.npy')
    pid = np.load('data/' + filename + '_pid.npy')
    label = np.load('data/' + filename + '_label.npy', allow_pickle=True)
    name = np.unique(pid)
    for i in name:
        Data[i] = data[pid == i]
        Label[i] = label[pid == i]
    return Data, Label


def create_sets():
    '''
    *** Description ***

    Create the sets for pre_training and fine_tune.
    Train means pre_training, and test means fine_tune.
    '''

    Data = []
    TmpLabel = []
    trainlen = 0
    files = ['cudb', 'mitdb', 'vfdb']
    for i in files:
        data, label = read_data(i)
        Data.extend(data.values())
        TmpLabel.extend(label.values())
    Label = []
    #encode Label
    for i in TmpLabel:
        Label.append(list(int('VF/VT' in j) for j in i))
    Label = get_time_to_sepsis(Label)
    Label = np.array(Label)
    # index = np.random.permutation(len(Label))
    # Data = Data[index]
    # Label = Label[index]

    for i in range(len(Data)):
        for j in range(len(Data[i])):
            tmp_data = Data[i][j]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            Data[i][j] = (tmp_data - tmp_mean) / tmp_std

    neg_data = list(Data[i][j] for i in range(len(Label)) for j in range(len(Label[i])) if Label[i][j][0] == 1)
    posi_data = list(Data[i][j] for i in range(len(Label)) for j in range(len(Label[i])) if Label[i][j][0] == 0)
    neg_label = list(Label[i][j] for i in range(len(Label)) for j in range(len(Label[i])) if Label[i][j][0] == 1)
    posi_label = list(Label[i][j] for i in range(len(Label)) for j in range(len(Label[i])) if Label[i][j][0] == 0)

    divide_propotion = 0.7
    neg_divide = int(len(neg_data) * divide_propotion)
    posi_divide = int(len(posi_label) * divide_propotion)
    train_data = neg_data[ : neg_divide] + posi_data[ : posi_divide]
    test_data = neg_data[neg_divide : ] + posi_data[posi_divide : ]
    train_label = neg_label[ : neg_divide] + posi_data[ : posi_divide]
    test_label = neg_label[neg_divide : ] + posi_data[posi_divide : ]


    return train_data, test_data, train_label, test_label

def bandpass_filter(data, lowcut, highcut, filter_order=4):
    nyquist_freq = 0.5 * data[1]['fs']

    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    for i in range(np.shape(data[0])[1]):
        data[0][:,i] = lfilter(b, a, data[0][:,i])
    return data

m = {'N': 'SN',  # Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
     'L': 'LBBB',  # Left bundle branch block beat
     'R': 'RBBB',  # Right bundle branch block beat
     'B': 'IVB',  # Bundle branch block beat (unspecified)
     'A': 'PAC',  # Atrial premature beat
     'a': 'PAC',  # Aberrated atrial premature beat
     'J': 'PJC',  # Nodal (junctional) premature beat
     'S': 'PSC',  # Supraventricular premature or ectopic beat (atrial or nodal)
     'V': 'PVC',  # Premature ventricular contraction
     'r': 'PVC',  # R-on-T premature ventricular contraction
     'F': 'PVC',  # Fusion of ventricular and normal beat
     'e': 'AE',  # Atrial escape beat
     'j': 'JE',  # Nodal (junctional) escape beat
     'n': 'SE',  # Supraventricular escape beat (atrial or nodal)
     'E': 'VE',  # Ventricular escape beat
     '/': 'PACED',  # Paced beat
     'f': 'PACED',  # Fusion of paced and normal beat
     'Q': 'OTHER',  # Unclassifiable beat
     '?': 'OTHER',  # Beat not classified during learning
     '[': 'VF',  # Start of ventricular flutter/fibrillation
     '!': 'VF',  # Ventricular flutter wave
     ']': 'VF',  # End of ventricular flutter/fibrillation
     'x': 'PAC',  # Non-conducted P-wave (blocked APC)
     '(AB': 'PAC',  # Atrial bigeminy
     '(AFIB': 'AF',  # Atrial fibrillation
     '(AF': 'AF',  # Atrial fibrillation
     '(AFL': 'AFL',  # Atrial flutter
     '(ASYS': 'PAUSE',  # asystole
     '(B': 'PVC',  # Ventricular bigeminy
     '(BI': 'AVBI',  # 1° heart block
     '(BII': 'AVBII',  # 2° heart block
     '(HGEA': 'PVC',  # high grade ventricular ectopic activity
     '(IVR': 'VE',  # Idioventricular rhythm
     '(N': 'SN',  # Normal sinus rhythm
     '(NOD': 'JE',  # Nodal (A-V junctional) rhythm
     '(P': 'PACED',  # Paced rhythm
     '(PM': 'PACED',  # Paced rhythm
     '(PREX': 'WPW',  # Pre-excitation (WPW)
     '(SBR': 'SNB',  # Sinus bradycardia
     '(SVTA': 'SVT',  # Supraventricular tachyarrhythmia
     '(T': 'PVC',  # Ventricular trigeminy
     '(VER': 'VE',  # ventricular escape rhythm
     '(VF': 'VF',  # Ventricular fibrillation
     '(VFL': 'VFL',  # Ventricular flutter
     '(VT': 'VT'  # Ventricular tachycardia
     }

labels = list(set(m.values()))


def resample_unequal(ts, fs_in, fs_out, t):
    """
    interploration
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=fs_in * t, endpoint=True)
        x_new = np.linspace(0, 1, num=fs_out * t, endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)

        return y_new


def get_label_map(labels):
    out_labels = []
    for i in labels:
        if i in m:
            if m[i] in ['VF', 'VFL', 'VT', '(VF', '(VT']:
                out_labels.append('VF/VT')
            else:
                out_labels.append('Others')
    out_labels = list(np.unique(out_labels))
    return out_labels

def preprocess_data(path, save_path, prefix):

    valid_lead = ['MLII', 'ECG', 'V5', 'V2', 'V1', 'ECG1', 'ECG2' ] # extract all similar leads

    t = 2
    window_size_t = 2 # second
    stride_t = 2 # second

    test_ind = []
    all_pid = []
    all_data = []
    all_label = []

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        cnt = 0
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print('read data failed')
            continue
        fs = tmp_data_res[1]['fs']
        window_size = int(fs*window_size_t)
        stride = int(fs*stride_t)
       # tmp_data_res = bandpass_filter(tmp_data_res, 0.5, 50)

        lead_in_data = tmp_data_res[1]['sig_name']
        print(lead_in_data)
        my_lead_all = []
        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)
        if len(my_lead_all) != 0:
            for my_lead in range(len(lead_in_data)):
                pp_pid = []
                pp_data = []
                pp_label = []
                channel = my_lead
                tmp_data = tmp_data_res[0][:, channel]
                idx_list = tmp_ann_res['sample']
                label_list = np.array(tmp_ann_res['symbol'])
                aux_list = np.array([i.strip('\x00') for i in tmp_ann_res['aux_note']])
                full_aux_list = [''] * tmp_data_res[1]['sig_len'] # expand aux to full length
                for i in range(len(aux_list)):
                    full_aux_list[idx_list[i]] = aux_list[i] # copy old aux
                    if label_list[i] in ['[', '!']:
                        full_aux_list[idx_list[i]] = '(VF' # copy VF start from beat labels
                    if label_list[i] in [']']:
                        full_aux_list[idx_list[i]] = '(N' # copy VF end from beat labels
                for i in range(1,len(full_aux_list)):
                    if full_aux_list[i] == '':
                        full_aux_list[i] = full_aux_list[i-1] # copy full_aux_list from itself, fill empty strings
                idx_start = 0

                while idx_start < len(tmp_data) - window_size:
                    idx_end = idx_start+window_size

                    tmpdata = resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out, t)
                    if not -100 < np.mean(tmpdata) < 100 or np.std(tmpdata) == 0:
                        idx_start += fs
                        continue
                    pp_pid.append("{}".format(record_name + '+' + str(my_lead)))

                    pp_data.append(resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out, t))
                    tmp_label_beat = label_list[np.logical_and(idx_list>=idx_start, idx_list<=idx_end)]
                    tmp_label_rhythm = full_aux_list[idx_start:idx_end] # be careful
                    tmp_label = list(np.unique(tmp_label_beat))+list(np.unique(tmp_label_rhythm))
                    tmp_label = get_label_map(tmp_label)
                    if 'VF/VT' in tmp_label and cnt <= 150:
                        idx_start += int(0.1 * fs)
                        cnt += 1
                    else:
                        idx_start += 2 * fs
                    pp_label.append(tmp_label)


                all_pid.extend(pp_pid)
                all_data.extend(pp_data)
                all_label.extend(pp_label)

                print('record_name:{}, len:{}, lead:{}, fs:{}, count:{}, labels:{}'.format(record_name, tmp_data_res[1]['sig_len'], my_lead, fs, len(pp_data), Counter(flatten(pp_label))))

        else:
            print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
            continue

    all_pid = np.array(all_pid)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print(all_pid.shape, all_data.shape)
    print(Counter(flatten(all_label)))
    print(Counter([tuple(_) for _ in all_label]))
    np.save(os.path.join(save_path, '{}_pid.npy'.format(prefix)), all_pid)
    np.save(os.path.join(save_path, '{}_data.npy'.format(prefix)), all_data)
    np.save(os.path.join(save_path, '{}_label.npy'.format(prefix)), all_label)
    print('{} done'.format(prefix))



if __name__ == "__main__":
    save_path = 'data/'
    source = True
    if not source:
        path = 'data/mit-bih-arrhythmia-database-1.0.0'
        prefix = 'mitdb'
        preprocess_data(path, save_path, prefix)

        path = 'data/mit-bih-malignant-ventricular-ectopy-database-1.0.0'
        prefix = 'vfdb'
        preprocess_data(path, save_path, prefix)

        path = 'data/cu-ventricular-tachyarrhythmia-database-1.0.0'
        prefix = 'cudb'
        preprocess_data(path, save_path, prefix)

        # path = 'data/mit-bih-normal-sinus-rhythm-database-1.0.0'
        # prefix = 'nsrdb'
        # preprocess_data(path, save_path, prefix)

    train_data, test_data, train_label, test_label = create_sets()
    np.save('data/train_data.npy', train_data)
    np.save('data/test_data.npy', test_data)
    np.save('data/train_label.npy', train_label)
    np.save('data/test_label.npy', test_label)