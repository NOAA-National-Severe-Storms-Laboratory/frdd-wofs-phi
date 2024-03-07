import numpy as np


path = "/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy"


fname = "wofs1d_20190430_1930_1954_v2030-2130.npy"


data = np.load("%s/%s" %(path, fname))

print (data.shape) 

idx = [i for i, arr in enumerate(data) if not np.isfinite(arr).all()]


print (idx) 
