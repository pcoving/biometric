import pandas as pd
import numpy as np
from time import time

def trim_mean(x, trim=10):
    lower_bound = np.percentile(x, trim)
    upper_bound = np.percentile(x, (100-trim))
    return np.mean(x[(x>=lower_bound) & (x<=upper_bound)])
    
SEED = 11

init = time(); print "Loading TRAINING file...",
data = pd.read_csv('data\\train.csv')[['T','Device']]
print "DONE! %.2fm" % ((time()-init)/60)

init = time(); print "Getting sample rate...",
## Create steps
data['T'] = data.groupby('Device').apply(lambda x: x['T'] - x['T'].shift(1)).fillna(207)
## Getting samples rate, then sort so that similar devices are close together
data2 = data.groupby('Device')['T'].apply(lambda x: trim_mean(x))
data2.sort()
print "DONE! %.2fm" % ((time()-init)/60)


## How many similar devices you'd want to include from the left and right
## Eg. If tol=3 you'll end up picking the next 3 devices with slightly higher
## samples rate AND the next 3 devices with slightly lower sample rate.
tol = 3

similars = []
for i,dev in enumerate(data2.index.values):
    begin=i-tol; end=i+tol; centre1 = i; centre2 = i
    if i < (tol+1):
        begin=0; centre2 = i+1; end = end+1
    if i > len(data2)-tol:
        end = len(data2); centre2 = i+1
    similars.append(
        (dev, list(data2.index.values[range(begin,centre1)]) +
         list(data2.index.values[range(centre2,end)]))
    )
    
similars = dict(similars)
print similars
