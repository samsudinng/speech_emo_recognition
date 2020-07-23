import torch.nn as nn
import torch
import torchvision

'''
2 Models Available:
   - SER_AlexNet   : AlexNet model from pyTorch
   - FCN+Attention : Fully-Convolutional model with AlexNet convolutional layer,
                        Attention layer, and output Linear classifier
'''

class Attention(nn.Module):
    '''
    Reference: "Attention Based Fully Convolutional Network for Speech Emotion Recognition"
    Authors: Yuanyuan Zhang and
             Jun Du and
             Zirui Wang and
             Jianshu Zhang and
             Yanhui Tu.
    '''

    def __init__(self, lambda_weight=0.3, n_channels=256):
        super(Attention, self).__init__()
        self.lambda_weight = torch.tensor(lambda_weight, dtype=torch.float32, requires_grad=False)
        self.linear1 = nn.Linear(in_features=n_channels, out_features=n_channels)
        self.linear2 = nn.Linear(in_features=n_channels, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        dims=x.size()
        x = x.reshape(dims[0],-1, dims[-1])
        
        v = self.tanh(self.linear1(x))
        v = self.linear2(v)
       
        v = v.squeeze(2)
        
        v = v*self.lambda_weight
        v = self.softmax(v)
        v = v.unsqueeze(-1)
        
        output = (x*v).sum(axis=1)
        
        
        return output
      

class SER_FCN_Attention(nn.Module):
    """
    Reference: "Attention Based Fully Convolutional Network for Speech Emotion Recognition"
    Authors: Yuanyuan Zhang and
             Jun Du and
             Zirui Wang and
             Jianshu Zhang and
             Yanhui Tu.

    Fully-Convolutional model with Attention. The convolution layer is taken from AlexNet.

    Parameters
    ----------
    num_classes : int
    in_ch       : int
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of the convolutional layer. 
        True initializes with AlexNet pre-trained weights.
    fcsize      : int
        The size of linear layer between Attention and output layer
    dropout     : float
        The dropout rate for linear layer between Attention and output layer
    
    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, dropout=0.2, in_ch=3, fcsize=256, pretrained=True):
        super(SER_FCN_Attention, self).__init__()

        #Use AlexNet from pyTorch
        alexnet = torchvision.models.alexnet(pretrained=pretrained)
        
        if in_ch != 3:
            alexnet.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        
        #FCN+Attention Convolution layer
        self.alexnet_conv = nn.Sequential(
                nn.BatchNorm2d(num_features=in_ch),
                alexnet.features[0],
                nn.BatchNorm2d(num_features=64),
                alexnet.features[1],
                alexnet.features[2],
                alexnet.features[3],
                nn.BatchNorm2d(num_features=192),
                alexnet.features[4],
                alexnet.features[5],
                alexnet.features[6],
                nn.BatchNorm2d(num_features=384),
                alexnet.features[7],
                alexnet.features[8],
                nn.BatchNorm2d(num_features=256),
                alexnet.features[9],
                alexnet.features[10],
                nn.BatchNorm2d(num_features=256),
                alexnet.features[11],
                alexnet.features[12],
                )
        

        #Attention layer
        self.attention = Attention(n_channels=256)
        
        #Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=fcsize),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=fcsize, out_features=num_classes))

        print('\n<< SER FCN_Attention model initialized >>\n')

    def forward(self, x):
        
        #Features layer
        x = self.alexnet_conv(x)
        
        #Attention layer
        x = x.permute([0,2,3,1]) #move channel to last dimension
        x = self.attention(x)
        
        #Classifier layer
        out = self.classifier(x)
        
        return out


class SER_AlexNet(nn.Module):
    """
    Reference: 
        https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch       : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet. 
        Set to 'True' for AlexNet pre-trained weights.
    fcsize      : int
        The size of output linear layer. Default AlexNet is 4096.
    
    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, in_ch=3, fcsize=4096, pretrained=True):
        super(SER_AlexNet, self).__init__()  

        model = torchvision.models.alexnet(pretrained=pretrained)
        model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        
        if fcsize != 4096:
            model.classifier[4] = nn.Linear(4096, fcsize)
            model.classifier[6] = nn.Linear(fcsize, num_classes)   
        else:
            model.classifier[6] = nn.Linear(4096, num_classes)   
        
        self.alexnet = model
        
        print('\n<< SER AlexNet model initialized >>\n')
    
    def forward(self, x):
        
        out = self.alexnet(x)

        return out


