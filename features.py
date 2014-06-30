import csv
from collections import defaultdict
import numpy as np
import marshal, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import warnings
import sys

# split up data in the same way it was prepared - i.e. test set comes from the end
# only compare to similar device during training? maybe not...

# add feature for percentage of discrete x,y,z values found in sample vs. full recording
# ^^^ might work better looking at set of deltas rather than absolute values.  also, combining x,y,z appears to work, as well as removing 0 and large values from the set

def calc_feature_dict(signal, slice=[0.,1.]):
    # XXXX should pass in fraction
    
    features = defaultdict(list)
    names = []
    
    # comptue magnitude
    signal = np.hstack([signal, np.sqrt(signal[:,1]*signal[:,1] + signal[:,2]*signal[:,2] + signal[:,3]*signal[:,3]).reshape(signal.shape[0],1)])

    percentiles = [0,10,20,30,40,50,60,70,80,90,100]
    for i in range(5):
        for p in percentiles: names.append('dx' + str(i) + '_' + str(p))
        features["raw"] += list(np.percentile(np.diff(signal[:,i]), percentiles))
        
    for p in percentiles: names.append('t_' + str(p))
    features["raw"] += list(np.percentile(signal[:,0]%(3600*24*1000.), percentiles))
    
    for i in range(1,5):
        for p in percentiles: names.append('x' + str(i) + '_' + str(p))
        features["raw"] += list(np.percentile(signal[:,i], percentiles))
    
    features['len'] = signal.shape[0]
    
    features["max_time"] = np.max(signal[:,0])
    features["min_time"] = np.min(signal[:,0])
    
    #signal = signal[signal.shape[0]*slice[0]:signal.shape[0]*slice[1]]    
    #accel_set = set(signal[:,1]).union(set(signal[:,2])).union(set(signal[:,3]))
    #features["diff_set"] = set(np.int32(np.diff(sorted(list(accel_set)))*1000))
    
    return names, features

def calc_features(device, sample, return_names=False):

    device_names, device_features = device
    sample_names, sample_features = sample
    
    names, features = [], []
    features += device_features['raw']
    features += sample_features['raw']

    ''''
    for dn, df in zip(device_names, device_features['raw']): 
        features.append(df)
        names.append('device_' + dn)
        
    for sn, sf in zip(sample_names, sample_features['raw']): 
        features.append(sf)
        names.append('sample_' + sn)
    '''
    '''
    diff_feat = []
    
    for dn, df, i in zip(device_names, device_features['raw'], range(len(device_features['raw']))): 
        for sn, sf, j in zip(sample_names, sample_features['raw'], range(len(sample_features['raw']))): 
            if (dn[0] == 't' and sn[0] == 't') or (dn[:3] == sn[:3]):
                if (int(dn.split('_')[1]) >= int(sn.split('_')[1])):
                    diff_feat.append((i,j))
                    features.append(df-sf)
                    names.append('diff_device_' + dn + '_sample_' + sn)

    marshal.dump(diff_feat, open('diff_feat.p', 'w'))
    '''
    diff_feat = marshal.load(open('diff_feat.p', 'r'))
    for i,j in diff_feat:
        features.append(device_features['raw'][i] - sample_features['raw'][j])

    names.append('device_len')
    features.append(device_features['len'])
                
    names.append('min_time-max_time')
    features += [sample_features["min_time"] - device_features["max_time"]]
    
    feat_imp = pickle.load(open('feat_imp.p', 'r'))    

    features = [features[idx] for idx in feat_imp[1:100]]

    '''
    names.append('diff_set_intersection')
    features += [float(len(sample_features["diff_set"].intersection(device_features["diff_set"])))/len(sample_features["diff_set"])]
    names.append('device_diff_set_len')
    features += [len(device_features["diff_set"])]
    names.append('sample_diff_set_len')
    features += [len(sample_features["diff_set"])]
    '''

    if return_names:
        return names, features
    else:
        return features

def parse_and_pickle():
    # pull out individual device signals from training data
    devices = []
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        reader.next() # skip header
        current_device = None
        for t, x, y, z, d in reader:
            if current_device != int(d):
                if current_device != None:
                    devices.append(current_device)
                    marshal.dump(signal, open('data/train/device' + str(current_device) + '.p', 'w'))
                current_device = int(d)
                print current_device
                signal = []
            signal += [[int(float(t)),float(x),float(y),float(z)]]
        
        # dump last one
        devices.append(current_device)
        marshal.dump(signal, open('data/train/device' + str(current_device) + '.p', 'w'))
    
    # save device identifiers
    devices = marshal.dump(devices, open('devices.p','w'))
    
    # pull out and pickle sequence samples from test data
    with open('test.csv', 'r') as f:
        reader = csv.reader(f)
        reader.next()
        current_device = None
        for t, x, y, z, d in reader:
            if current_device != int(d):
                if current_device != None:
                    marshal.dump(signal, open('data/test/sequence' + str(current_device) + '.p', 'w'))
                current_device = int(d)            
                print current_device
                signal = []
            signal += [[int(float(t)),float(x),float(y),float(z)]]
        marshal.dump(signal, open('data/test/sequence' + str(current_device) + '.p', 'w'))
        
def compute_features(n_other_devices=5, train_ratio=0.7, use_traces=True):

    print 'computing device features...'
    devices = marshal.load(open('devices.p','r'))
    
    min_train_time, max_train_time = 1e20, -1e20
    device_features = {}
    for device in devices:
        print '.',
        sys.stdout.flush()
        # XXXXX here only take first <train_ratio> or so of data
        # XXXX some features should be computed on WHOLE signal (min/max T for example) and some on subset to prevent leakage (intersection of delta sets for example)
        signal = np.asarray(marshal.load(open('data/train/device' + str(device) + '.p', 'r')))
        #signal = signal[:(signal.shape[0]*train_ratio)]
        max_train_time = max(np.max(signal[:,0]), max_train_time)
        min_train_time = min(np.min(signal[:,0]), min_train_time)
        device_features[device] = calc_feature_dict(signal) #, slice=[0.,train_ratio])
    
    #print 'min train time:', min_train_time
    #print 'max train time:', max_train_time
    
    min_test_time, max_test_time = 1e20, -1e20
    print 'building training set...'        
    train_features, train_labels = [], []
    for device in devices:
        print '.',
        sys.stdout.flush()
        tmp_devices = devices[:]
        tmp_devices.remove(device)
        # XXXX here only take last <1.0-train_ratio> or so of features
        signal = np.asarray(marshal.load(open('data/train/device' + str(device) + '.p', 'r')))
        
        # XXXX should preserve time of day (T%86000...) 
        signal[:,0] = signal[:,0] - signal[0,0] + signal[0,0]%(24*3600*1000.) + (np.floor(max_train_time/(24*3600*1000.)) + 1.)*24*3600*1000.
        min_test_time = min(np.min(signal[:,0]), min_test_time)
        max_test_time = max(np.max(signal[:,0]), max_test_time)

        #signal = signal[(signal.shape[0]*train_ratio):]
        
        # chop up into length-300 size snippets
        
        for idx0, idx1 in zip(range(0,signal.shape[0],300), range(300,signal.shape[0],300)):
            sample_features = calc_feature_dict(signal[idx0:idx1])
            train_labels.append(1)
            train_features.append(calc_features(device=device_features[device], sample=sample_features))
            
            # choose n other devices for "false" samples
            other_devices = np.random.choice(tmp_devices, size=n_other_devices, replace=False)
            for other_device in other_devices:
                train_labels.append(0)
                train_features.append(calc_features(device=device_features[other_device], sample=sample_features))

    #print 'min test time:', min_test_time    
    #print 'max test time:', max_test_time

    #marshal.dump(train_features, open('data/train/train_features.p', 'w'))
    #marshal.dump(train_labels, open('data/train/train_labels.p', 'w'))
    
    if (use_traces):
        traces = marshal.load(open('data/train/traces.p','r'))

        trace_idx = {}
        for idx, lt in enumerate(traces):
            for elem in lt:
                trace_idx[elem] = idx
    
        seq_features = {}

        print 'generating question features...'
        question_features, question_ids = [], []
        with open('questions.csv', 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for qid, sid, device in reader:
                print qid
                if int(sid) in trace_idx:
                    for sid2 in traces[trace_idx[int(sid)]]:
                        question_ids.append(int(qid))
                        if sid2 not in seq_features:
                            signal = np.asarray(marshal.load(open('data/test/sequence' + str(sid2) + '.p', 'r')))
                            seq_features[sid2] = calc_feature_dict(signal)
                            question_features.append(calc_features(device=device_features[int(device)], sample=seq_features[sid2]))
                        else:
                            question_features.append(calc_features(device=device_features[int(device)], sample=seq_features[sid2]))
                else:
                    question_ids.append(int(qid))
                    signal = np.asarray(marshal.load(open('data/test/sequence' + sid + '.p', 'r')))
                    question_features.append(calc_features(device=device_features[int(device)], sample=calc_feature_dict(signal)))
    else:    
        print 'generating question features...'
        question_features, question_ids = [], []
        with open('questions.csv', 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for qid, sid, device in reader:
                #print '.', 
                #sys.stdout.flush()
                question_ids.append(int(qid))
                signal = np.asarray(marshal.load(open('data/test/sequence' + sid + '.p', 'r')))
                question_features.append(calc_features(device=device_features[int(device)], sample=calc_feature_dict(signal)))

    
    #marshal.dump(question_features, open('data/test/question_features.p', 'w'))
    #marshal.dump(question_ids, open('data/test/question_ids.p', 'w'))

    return train_features, train_labels, question_features, question_ids

def match_traces():

    print 'matching traces...'

    start_times, end_times, end_accel, start_accel = [], [], [], []
    sequences = []
    with open('questions.csv', 'r') as f:
        reader = csv.reader(f)
        reader.next()
        for qid, sid, device in reader:
            signal = np.asarray(marshal.load(open('data/test/sequence' + sid + '.p', 'r')))
            start_times.append(signal[0,0])
            end_times.append(signal[-1,0])
            start_accel.append(signal[0,1:])
            end_accel.append(signal[-1,1:])
            sequences.append(sid)
    
    idx_pairs = {}
    
    end_times_idx = sorted(zip(end_times, range(len(end_times))))
    start_times_idx = sorted(zip(start_times, range(len(end_times))))
    
    count = 0
    for t1, idx1 in end_times_idx:
        myiter = (val for val in start_times_idx if val[0] > t1)
        try:
            t2, idx2 = next(myiter)
            while t2-t1 < 300:    
                if idx1 not in idx_pairs:
                    diff = ((start_accel[idx2][0] - end_accel[idx1][0])**2 +
                            (start_accel[idx2][1] - end_accel[idx1][1])**2 +
                            (start_accel[idx2][2] - end_accel[idx1][2])**2)
                    idx_pairs[idx1] = (idx2, diff)
                else:
                    diff = ((start_accel[idx2][0] - end_accel[idx1][0])**2 +
                            (start_accel[idx2][1] - end_accel[idx1][1])**2 +
                            (start_accel[idx2][2] - end_accel[idx1][2])**2)
                    if diff < idx_pairs[idx1][1]:
                        print sequences[idx1], sequences[idx_pairs[idx1][0]], sequences[idx2], diff, idx_pairs[idx1][1]
                        idx_pairs[idx1] = (idx2, diff)
                t2, idx2 = next(myiter)
                count += 1
                if count%1000 == 0:
                    print count
        except StopIteration:
            pass
    
    traces = defaultdict(list)
    
    for t, idx in start_times_idx:
        if idx in idx_pairs:
            current_idx = idx
            try:
                while True:
                    traces[idx] += [idx_pairs[current_idx][0]]
                    next_idx = idx_pairs[current_idx][0]
                    del idx_pairs[current_idx]
                    current_idx = next_idx
            except KeyError:
                pass
    
    traces_new = []
    for k,v in traces.items():
        traces_new.append([int(sequences[k])] + [int(sequences[x]) for x in v])

    marshal.dump(traces_new, open('data/train/traces.p', 'w'))
    
def make_submission(answers, question_ids):
    print 'making submission...'
    
    with open('submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['QuestionId','IsTrue'])
        
        current_id = 1
        vals = []
        for idx, val in zip(question_ids, answers[:,1]):
            if idx != current_id:
                #print idx, current_id
                assert idx == current_id+1
                writer.writerow([current_id, np.mean(vals)])
                vals = []
                current_id = idx

            vals += [val]

        writer.writerow([current_id, np.mean(vals)])

if __name__ == "__main__":
    
    #parse_and_pickle()
    #match_traces()    

    warnings.simplefilter('ignore')
    recompute_features = True

    if (recompute_features):
        np.random.seed(1)
        train_features, train_labels, question_features, question_ids = compute_features(n_other_devices=5, use_traces=True)
    else:
        assert(0)
        train_features, train_labels, question_features, question_ids = marshal.load(open('data/train/features.p','r'))
    
    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)
    question_features = np.asarray(question_features)
    question_ids = np.asarray(question_ids)

    classifier = RandomForestClassifier(verbose=2, n_estimators=100, n_jobs=1)
    #classifier = GradientBoostingClassifier(verbose=2, n_estimators=100)
    #classifier = DecisionTreeClassifier(max_depth=5)
    
    classifier.fit(train_features, train_labels)
    answers = classifier.predict_proba(question_features)
    
    make_submission(answers, question_ids)

