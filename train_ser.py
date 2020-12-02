import sys
import argparse
import pickle
from data_utils import SERDataset
import torch
import numpy as np
from model import SER_AlexNet, SER_AlexNet_GAP
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os
import random
from collections import Counter
from torch.backends import cudnn


def main(args):
    
    # Aggregate parameters
    params={
            #model & features parameters
            'ser_model': args.ser_model,

            #training
            'val_id': args.val_id,
            'test_id': args.test_id,
            'num_epochs':args.num_epochs,
            'batch_size':args.batch_size,
            'lr':args.lr,
            'random_seed':args.seed,
            'use_gpu':args.gpu,
            
            #best mode
            'save_label': args.save_label,
            
            #parameters for tuning
            'oversampling': args.oversampling,
            'pretrained': args.pretrained,
            'mixup' : args.mixup
            }

    print('*'*40)
    print(f"\nPARAMETERS:\n")
    print('*'*40)
    print('\n')
    for key in params:
        print(f'{key:>15}: {params[key]}')
    print('*'*40)
    print('\n')

    #set random seed
    seed_everything(params['random_seed'])

    # Load dataset
    with open(args.features_file, "rb") as fin:
        features_data = pickle.load(fin)

    ser_dataset = SERDataset(features_data,
                               val_speaker_id=args.val_id,
                               test_speaker_id=args.test_id,
                               oversample=args.oversampling
                               )

    # Train
    train_stat = train(ser_dataset, params, save_label=args.save_label)

    return train_stat


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a SER  model in an iterative-based manner with "
                    "pyTorch and IEMOCAP dataset.")

    #Features
    parser.add_argument('features_file', type=str,
        help='Features extracted from `extract_features.py`.')
    
    #Model
    parser.add_argument('--ser_model', type=str, default='ser_alexnet',
        help='SER model to be loaded')
    
    #Training
    parser.add_argument('--val_id', type=str, default='1F',
        help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default='1M',
        help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=200,
        help='Number of training epochs.') 
    parser.add_argument('--batch_size', type=int, default=32,
        help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=100,
        help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=1,
        help='If 1, use GPU')
    
    #Best Model
    parser.add_argument('--save_label', type=str, default=None,
        help='Label for the current run, used to save the best model ')

    #Parameters for model tuning
    parser.add_argument('--oversampling', action='store_true',
        help='By default, no oversampling is applied to training dataset.'
             'Set this to true to apply random oversampling to balance training dataset')
    
    parser.add_argument('--pretrained', action='store_true',
        help='By default, SER_AlexNet or SER_AlexNet_GAP model weights are'
             'initialized randomly. Set this flag to initalize with '
             'ImageNet pre-trained weights.')
    
    parser.add_argument('--mixup', action='store_true',
        help='Set this to true to perform mixup at dataloader')

    return parser.parse_args(argv)



def test(model, criterion, test_dataset, batch_size, device,
         return_matrix=False):

    """Test an SER model.

    Parameters
    ----------
    model
        PyTorch model
    criterion
        loss_function
    test_dataset
        The test dataset
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    loss, weighted accuracy (WA), unweighted accuracy (UA), confusion matrix 
       

    """
    total_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    for i, batch in enumerate(test_loader):
        test_data_batch, test_labels_batch = batch

        # Send to correct device
        test_data_batch = test_data_batch.to(device)
        test_labels_batch = test_labels_batch.to(device, dtype=torch.long)
        
        # Forward
        test_preds_batch = model(test_data_batch)
        test_preds_segs.append(f.log_softmax(test_preds_batch, dim=1).cpu())
        
        #test loss
        test_loss = criterion(test_preds_batch, test_labels_batch)
        total_loss += test_loss.item()

    # Average loss
    test_loss = total_loss / (i+1)

    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = test_dataset.get_preds(test_preds_segs)
    
    # Make sure everything works properly
    assert len(test_preds) == test_dataset.n_actual_samples
    test_wa = test_dataset.weighted_accuracy(test_preds)
    test_ua = test_dataset.unweighted_accuracy(test_preds)

    results = (test_loss, test_wa*100, test_ua*100)
    
    if return_matrix:
        test_conf = test_dataset.confusion_matrix_iemocap(test_preds)
        return results, test_conf
    else:
        return results
    


def train(dataset, params, save_label='default'):

    #get dataset
    train_dataset = dataset.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=True)

    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()
    
    #select device
    if params['use_gpu'] == 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Construct model, optimizer and criterion
    pretrained  = params['pretrained']
    ser_model   = params['ser_model']
    num_classes = dataset.num_classes
    num_in_ch   = dataset.num_in_ch
    mixup = params['mixup']    
    batch_size = params['batch_size']
    
    if ser_model == 'alexnet':
        model = SER_AlexNet(num_classes=num_classes,
                            in_ch=num_in_ch,
                            pretrained=pretrained).to(device)
    elif ser_model == 'alexnet_gap':
        model = SER_AlexNet_GAP(num_classes=num_classes,
                            in_ch=num_in_ch,
                            pretrained=pretrained).to(device)
    else:
        raise ValueError('No model found!')
    
    
    print(model.eval())
    print(f"Number of trainable parameters: {count_parameters(model.train())}")
    print('\n')

    #Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    if mixup == True:
        criterion_mixup = nn.CrossEntropyLoss(reduction='none')

    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    acc_format2 = "{:.02f}"
    best_val_wa = 0
    best_val_ua = 0
    save_path = save_label + '.pth'

    all_train_loss =[]
    all_train_wa =[]
    all_train_ua=[]
    all_val_loss=[]
    all_val_wa=[]
    all_val_ua=[]
    
    for epoch in range(params['num_epochs']):
        
        #get current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Train one epoch
        total_loss = 0
        train_preds = []
        target=[]
        model.train()
        for i, batch in enumerate(train_loader):
            
            train_data_batch, train_labels_batch = batch

            # Clear gradients
            optimizer.zero_grad()
            
            # Send data to correct device
            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device,dtype=torch.long)
            
            
            if mixup == True:
                # Mixup
                inputs, targets_a, targets_b, lam = mixup_data_enh(train_data_batch, 
                        train_labels_batch, alpha= 0.4, use_cuda=torch.cuda.is_available())
                
                lam = lam.to(device)

                # Forward pass
                preds = model(inputs)

                # Loss
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                train_loss = loss_func(criterion_mixup, preds)
                train_loss = torch.mean(train_loss)

            else:
                # Forward pass
                preds = model(train_data_batch)

                # Loss
                train_loss = criterion(preds, train_labels_batch)
            
            # Compute the loss, gradients, and update the parameters
            total_loss += train_loss.detach().item()
            train_loss.backward()
            optimizer.step()
            
        # Evaluate training data
        train_loss = total_loss / (i+1)
        all_train_loss.append(loss_format.format(train_loss))
        
        
        #Validation
        with torch.no_grad():
            val_result = test(
                model, criterion, val_dataset, 
                batch_size=64, 
                device=device)
        
            val_loss = val_result[0]
            val_wa = val_result[1]
            val_ua = val_result[2]

            # Update best model based on validation UA
            if val_wa > best_val_wa:
                best_val_ua = val_ua
                best_val_wa = val_wa
                best_val_loss = val_loss
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)

        all_val_loss.append(loss_format.format(val_loss))
        all_val_wa.append(acc_format2.format(val_wa))
        all_val_ua.append(acc_format2.format(val_ua))
        
        print(f"Epoch {epoch+1}  (lr = {current_lr})\
        	Loss: {loss_format.format(train_loss)} - {loss_format.format(val_loss)} - WA: {acc_format.format(val_wa)} <{acc_format.format(best_val_wa)}> - UA: {acc_format.format(val_ua)} <{acc_format.format(best_val_ua)}>")

    # Test on best model
    with torch.no_grad():
        model.load_state_dict(torch.load(save_path))

        test_result, confusion_matrix = test(
            model, criterion, test_dataset, 
            batch_size=1, #params['batch_size'],
            device=device, return_matrix=True)

        print("*" * 40)
        print("RESULTS ON TEST SET:")
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: "
              "{:.2f}".format(test_result[0], test_result[1], test_result[2]))
        print("Confusion matrix:\n{}".format(confusion_matrix[1]))   
        

    return(all_train_loss, all_train_wa, all_train_ua,
            all_val_loss, all_val_wa, all_val_ua,
            loss_format.format(test_result[0]), 
            acc_format2.format(test_result[1]),
            acc_format2.format(test_result[2]),
            confusion_matrix[0])


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark=True
    cudnn.deterministic = True


# to count the number of trainable parameter in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#mixup augmentation
def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


#enhanced version of mixup augmentation
def mixup_data_enh(x, y, alpha=0.4, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
            Enhancements:
            1. lambda = max(lambda, 1-lambda)
            2. draw different lambda for each samples in the batch   
    '''
    batch_size = x.size()[0]    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    x_shuffled = x[index,:]

    
    mixed_x=x+x_shuffled #just to initialize
    lam_batch=torch.empty(batch_size)
    for i,(xo,xs) in enumerate(list(zip(x,x_shuffled))):

        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        lam_batch[i] = lam
        
        #mix = (lam * xo + (1 - lam) * xs).unsqueeze(0)
        mixed_x[i] = (lam * xo + (1 - lam) * xs)#torch.cat((mixed_x,mix),0)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam_batch

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
