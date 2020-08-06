import pickle
import numpy as np 
import pandas as pd

exp_no      = 1
kfold       = 5
num_classes = 4

datadir  = 'SER_Repo/results/'
result_id= {'1':['allstat_iemocap_fcn_attention_0.pkl']
            }


test_loss = []
test_wa=[]
test_ua=[]
test_conf=np.zeros((num_classes,num_classes),dtype=int)
test_conf_avg=np.zeros((num_classes,num_classes),dtype=float)
n_files=0

for f in result_id[str(exp_no)]:
    try:
        with open(datadir+f, "rb") as fin:
            all_stat = pickle.load(fin)
        n_files += 1
    except:
        continue

    numfold = len(all_stat)
    incr=1
    if kfold == 5 and numfold == 10:
        incr=2

    for i in range(0,numfold,incr):
        
        if kfold == 5 and numfold == 10:
            #use best of the 2 WA for each session
            imax = np.argmax(np.array([all_stat[i][7], all_stat[i+1][7]]))
            imax += i
            test_loss.append(all_stat[imax][6])
            test_wa.append(all_stat[imax][7])
            test_ua.append(all_stat[imax][8])
            test_conf += np.array(all_stat[imax][9])
        else:
            test_loss.append(all_stat[i][6])
            test_wa.append(all_stat[i][7])
            test_ua.append(all_stat[i][8])
            test_conf += np.array(all_stat[i][9])

    
classes_total = test_conf.sum(axis=1).transpose()
classes_total = np.column_stack((classes_total,classes_total,classes_total,classes_total))
test_conf_avg += (test_conf/classes_total)*100
    

print(result_id[str(exp_no)][n_files-1])
print(f'Num. fold: {len(test_loss)} ({n_files} sessions)')
print(f'>> WA <<\nAvg: {np.array(test_wa,dtype=float).mean()}\nStd: {np.array(test_wa,dtype=float).std()}\n{test_wa}\n')
print(f'>> UA <<\nAvg: {np.array(test_ua,dtype=float).mean()}\nStd: {np.array(test_ua,dtype=float).std()}\n{test_ua}\n')
print(f'>> LOSS << {test_loss}\n')


test_conf = test_conf_avg #/len(result_feature)
conf_format = "{:.02f}%"

test_conf_fmt = pd.DataFrame({"ang": [conf_format.format(p) for p in test_conf[:,0]], "sad": [conf_format.format(p) for p in test_conf[:,1]],
                             "hap": [conf_format.format(p) for p in test_conf[:,2]], "neu": [conf_format.format(p) for p in test_conf[:,3]]})

test_conf_fmt = test_conf_fmt.to_string(index=False)
print(test_conf_fmt)

