import pickle
import numpy as np 
import pandas as pd
from collections import OrderedDict
import sys
import argparse

def main(args):

    label         = args.label
    kfold         = args.kfold
    n_runs        = args.n_runs
    datadir       = args.path

    num_classes   = 4
    num_sessions  = 5
    classes       = ["ang", "sad","hap","neu"]
    test_id       = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F']
    sessions      = ['Sess.1','Sess.2','Sess.3','Sess.4','Sess.5']
    five_fold_idx = [0,3,5,7,9] #index of speaker ID used as test_id for 5 fold CV -> [1M, 2F, 3M, 4F, 5F]
    conf_matrix_display = 'percentage' # 'count' or 'percentage'


    """
    Format of the result file (allstat*): [(fold1_data), (fold2_data) ..., (fold10_data)]
    Each foldx_data is tuple:
	[0] all epoch train loss
	[1] all epoch train wa
	[2] all epoch train ua
        [3] all epoch val loss
        [4] all epoch val wa
        [5] all epoch val ua
	[6] test loss
	[7] test WA
	[8] test UA
	[9] test confusion matrix	
     """

    # CALCULATE RESULTS

    test_loss = []
    test_wa=[]
    test_ua=[]
    test_conf=np.zeros((num_classes,num_classes),dtype=int)
    test_conf_avg=np.zeros((num_classes,num_classes),dtype=float)
    n_files = 0

    for ns in range(n_runs):
        f = f'allstat_iemocap_{label}_{ns}.pkl'
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
                if i in five_fold_idx:
                    idx = i
                else:
                    idx = i+1
                
                test_loss.append(all_stat[idx][6])
                test_wa.append(all_stat[idx][7])
                test_ua.append(all_stat[idx][8])
                test_conf += np.array(all_stat[idx][9])
            else:
                test_loss.append(all_stat[i][6])
                test_wa.append(all_stat[i][7])
                test_ua.append(all_stat[i][8])
                test_conf += np.array(all_stat[i][9])


    # Calculate percentage for confusion matrix   
    classes_total = np.expand_dims(test_conf.sum(axis=1).transpose(), axis=1)
    classes_total = np.repeat(classes_total,num_classes, axis=1)
    test_conf_avg += (test_conf/classes_total)*100

    #convert metrics to data frame for easy printing
    avg_wa_run = np.array(test_wa,dtype=float).reshape(n_files,-1).mean(axis=1)
    avg_ua_run = np.array(test_ua,dtype=float).reshape(n_files,-1).mean(axis=1)
    std_wa_run = np.array(test_wa,dtype=float).reshape(n_files,-1).std(axis=1)
    std_ua_run = np.array(test_ua,dtype=float).reshape(n_files,-1).std(axis=1)

    test_wa = np.array(test_wa).reshape(n_files, num_sessions, -1)
    test_ua = np.array(test_ua).reshape(n_files, num_sessions, -1)
    test_loss = np.array(test_loss).reshape(n_files, num_sessions, -1)    



    print(f"\n{kfold}-fold Cross Validation x {n_files} runs")
    print(f'Cross-val Label: {label}')
    print(f"     {'WA':>8}{'UA':>8}")
    print(f'Avg: {round(np.array(test_wa,dtype=float).mean(),1):8}{round(np.array(test_ua,dtype=float).mean(),1):8}')      
    print(f'Std: {round(np.array(test_wa,dtype=float).std(),1):8}{round(np.array(test_ua,dtype=float).std(),1):8}') 
    print(f'Min: {round(np.array(test_wa,dtype=float).min(),1):8}{round(np.array(test_ua,dtype=float).min(),1):8}') 
    print(f'Max: {round(np.array(test_wa,dtype=float).max(),1):8}{round(np.array(test_ua,dtype=float).max(),1):8}')      

    dict_data = OrderedDict()
    for nf in range(n_files):
        dict_data['Run '+ str(nf)] = ['','','','','','','']
        fwa = [list(wa) for wa in test_wa[nf]]
        fwa.extend([round(avg_wa_run[nf],1), round(std_wa_run[nf],1)]) 
        fua = [list(ua) for ua in test_ua[nf]]
        fua.extend([round(avg_ua_run[nf],1), round(std_ua_run[nf],1)])
        floss = [list(loss) for loss in test_loss[nf]]
        floss.extend(['',''])
      
        dict_data[f'[{str(nf+1)}] WA'] = fwa
        dict_data[f'[{str(nf+1)}] UA'] = fua
        dict_data[f'[{str(nf+1)}] LOSS'] = floss

    columns = [s+' [M,F]' for s in sessions]
    columns.extend(['--> Mean', 'Std'])

    df = pd.DataFrame.from_dict(dict_data, orient='index', columns=columns)

    print('\n')
    print(df.to_string())


    if conf_matrix_display == 'percentage':
        test_conf = test_conf_avg #/len(result_feature)
        conf_format = "{:.02f}%"
    elif conf_matrix_display == 'count':
        conf_format = "{}"
    else:
        raise valueError('Wrong format!!')

    test_conf_df = OrderedDict()
    for i, c in enumerate(classes):
        test_conf_df[c] = [conf_format.format(p) for p in test_conf[:,i]]
    test_conf_df = pd.DataFrame(test_conf_df, index=classes)

    print('\n')
    print(test_conf_df.to_string(index=True))




def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get the result for k-fold Cross-Validation ")

    parser.add_argument('label', type=str,
        help='Cross-validation label')

    parser.add_argument('kfold', type=int,
        help='Number of folds (5 or 10)')

    parser.add_argument('n_runs', type=int,
        help='Number of k-fold repeats')

    parser.add_argument('path', type=str,
        help='kfold results file path')
    
    return parser.parse_args(argv)



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
