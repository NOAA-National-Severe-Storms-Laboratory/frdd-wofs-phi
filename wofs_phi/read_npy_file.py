import numpy as np
import pandas as pd 

#path = "/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy"


#fname = "wofs1d_20190430_1930_1954_v2030-2130.npy"
#fnames = np.genfromtxt("test_files.txt", dtype='str') 
#fnames = np.genfromtxt("/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/training_filenames.txt",\
#                        dtype='str') 

fnames = ["/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/test_train_data.npy"]
names = np.genfromtxt("rf_variables.txt", dtype='str') 

for fname in fnames:
    print (fname) 


    #data = np.load("%s/%s" %(path, fname))
    data = np.load(fname) 

    #Read this into pandas dataframe

    df = pd.DataFrame(data, columns=names)

    #replace infinite values with nans 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_rows = df.isna().any(axis=1)
    nan_cols = df.isna().any(axis=0)

    print (names[nan_cols])
    print (sum(nan_rows))

    #print (nan_rows)
    #print (nan_cols) 
    print ("*****") 



