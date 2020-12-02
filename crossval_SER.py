import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os

repeat_kfold = 5 # to  perform k-fold for n-times with different seed

#------------PARAMETERS---------------#

features_file = 'IEMOCAP_logspec200.pkl'
ser_model     = 'alexnet_gap'

val_id =  ['1F','1M','2F','2M','3F','3M','4F','4M','5F','5M']
test_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F']
num_epochs  = '100'
batch_size  = '64'
lr          = '0.0001'
random_seed = 111
gpu = '1'
save_label = 'fcn_mixup'

#additional flags: --pretrained, --mixup, --oversampling should be added directly
#   into train_ser.sys.argv as shown below




#Start Cross Validation

for repeat in range(repeat_kfold):

    all_stat = []
    random_seed +=  (repeat*100)
    seed = str(random_seed)

    for v_id, t_id in list(zip(val_id, test_id)):

        train_ser.sys.argv      = [
                        
                                  'train_ser.py', 
                                  features_file,
                                  '--ser_model',ser_model,
                                  '--val_id',v_id, 
                                  '--test_id', t_id,
                                  '--gpu', gpu,
                                  '--num_epochs', num_epochs,
                                  '--batch_size', batch_size,
                                  '--lr', lr,
                                  '--seed', seed,
                                  '--save_label', save_label,
                                  '--pretrained',
                                  '--mixup'
                                  ]

    
        stat = train_ser.main(parse_arguments(sys.argv[1:]))   
        all_stat.append(stat)       
        os.remove(save_label+'.pth')
    


    with open('allstat_iemocap_'+save_label+'_'+str(repeat)+'.pkl', "wb") as fout:
        pickle.dump(all_stat, fout)
