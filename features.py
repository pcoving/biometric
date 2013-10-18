import csv
from collections import defaultdict
import numpy as np
import marshal
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict

def calc_features(signal):
    
    features = []
    '''
    for i in range(4):
        features += [np.max(signal[:,i])]
        features += [np.min(signal[:,i])]
        features += [np.max(np.diff(signal[:,i]))]
        features += [np.min(np.diff(signal[:,i]))]
    '''
    dt = np.diff(signal[:,0])
    for percentile in [10,20,30,40,50,60,70,80,90]: #[25, 50, 75]:
        features += [np.percentile(dt, percentile)]
        
        for i in range(3):
            features += [np.percentile(signal[:,i+1], percentile)]
            #features += [scipy.stats.scoreatpercentile(np.diff(signal[:,i+1])/np.maximum(dt, np.ones(dt.size)), percentile)]

    features += [signal.shape[0]]
    
    return features

def parse_and_pickle():
    devices = []
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        reader.next()
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
        
        devices.append(current_device)
        marshal.dump(signal, open('data/train/device' + str(current_device) + '.p', 'w'))
    
    devices = marshal.dump(devices, open('devices.p','w'))

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

def compute_features(n_other_devices=5):

    print 'computing device features...'
    devices = marshal.load(open('devices.p','r'))
    # compute features on entire signal
    device_features = {}
    for device in devices:
        print device
        signal = np.asarray(marshal.load(open('data/train/device' + str(device) + '.p', 'r')))
        device_features[device] = calc_features(signal)

    print 'building training set...'        
    train_features = []
    train_labels = []
    for device in devices:
        print device
        tmp_devices = devices[:]
        tmp_devices.remove(device)
        signal = np.asarray(marshal.load(open('data/train/device' + str(device) + '.p', 'r')))
        for idx0, idx1 in zip(range(0,signal.shape[0],300), range(300,signal.shape[0],300)):
            features = calc_features(signal[idx0:idx1])
            train_labels.append(1)
            train_features += [device_features[device] + features]
        
            other_devices = np.random.choice(tmp_devices, size=n_other_devices, replace=False)
            for other_device in other_devices:
                train_labels.append(0)
                train_features += [device_features[other_device] + features]
        
    #marshal.dump(train_features, open('data/train/train_features.p', 'w'))
    #marshal.dump(train_labels, open('data/train/train_labels.p', 'w'))

    traces = marshal.load(open('data/train/traces.p','r'))
    
    trace_idx = {}
    for idx, lt in enumerate(traces):
        for elem in lt:
            trace_idx[elem] = idx
    
    seq_features = {}

    print 'generating question features...'
    question_features = []
    question_ids = []
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
                        seq_features[sid2] = calc_features(signal)
                        question_features.append(device_features[int(device)] + seq_features[sid2])
                    else:
                        question_features.append(device_features[int(device)] + seq_features[sid2])
            else:
                question_ids.append(int(qid))
                signal = np.asarray(marshal.load(open('data/test/sequence' + sid + '.p', 'r')))
                question_features.append(device_features[int(device)] + calc_features(signal))
            
    #marshal.dump(question_features, open('data/test/question_features.p', 'w'))
    #marshal.dump(question_ids, open('data/test/question_ids.p', 'w'))

    return train_features, train_labels, question_features, question_ids
    
def make_submission(answers, question_ids):
    print 'making submission...'
    
    with open('submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['QuestionId','IsTrue'])
        
        current_id = 1
        numer, denom = 0., 0.
        for idx, val in zip(question_ids, answers[:,1]):
            if idx != current_id:
                print idx, current_id
                assert idx == current_id+1
                writer.writerow([current_id, numer/denom])
                numer, denom = 0., 0.
                current_id = idx
            
            numer += val
            denom += 1.

        writer.writerow([current_id, numer/denom])
            
if __name__ == "__main__":
    
    #parse_and_pickle()
    '''
    recompute_features = True

    if (recompute_features):
        np.random.seed(1)
        train_features, train_labels, question_features, question_ids = compute_features()
    else:
        assert(0)
        train_features, train_labels, question_features, question_ids = marshal.load(open('data/train/features.p','r'))

    train_features, train_labels, question_features, question_ids = (np.asarray(train_features), 
                                                                     np.asarray(train_labels),
                                                                     np.asarray(question_features),
                                                                     np.asarray(question_ids))
    '''
    classifier = RandomForestClassifier(verbose=2, n_estimators=200, n_jobs=1)
    #classifier = GradientBoostingClassifier(verbose=2, n_estimators=100)
    
    classifier.fit(train_features, train_labels)
    answers = classifier.predict_proba(question_features)
    
    make_submission(answers, question_ids)

'''
start_times = []
end_times = []
end_accel = []
start_accel = []
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
                    #print sequences[idx1], sequences[idx_pairs[idx1][0]], sequences[idx2], diff, idx_pairs[idx1][1]
                    idx_pairs[idx1] = (idx2, diff)
            t2, idx2 = next(myiter)
        count += 1
        if count%1000 == 0:
            print count
    except StopIteration:
        pass

idx_pairs = idx_pairs_copy.copy()
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
'''

