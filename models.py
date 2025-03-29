import torch
import torch.nn as nn
import torch.nn.functional as F

class R_Encoder(nn.Module):
    """
    Relational encoder.
    
    Args:
        kernel_size (int): The kernel size for the first convolution layer.
        kernel_size2 (int): The kernel size for the second convolution layer.
        max_kernel_size (int): The maximum kernel size for the first convolution.
        max_kernel_size2 (int): The maximum kernel size for the second convolution.

    Attributes:
        out_channels (int): Number of output channels for the second convolution.
        feat_dim (int): Dimensionality of the output features.
    """        
    def __init__(self, kernel_size, kernel_size2, max_kernel_size, max_kernel_size2):
        super(R_Encoder, self).__init__()
        self.out_channels = 64
        self.feat_dim = (max_kernel_size2-kernel_size2+1)*self.out_channels

        self.trans = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(kernel_size,1)),
                                   nn.BatchNorm2d(32, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(max_kernel_size-kernel_size+1,kernel_size2)),
                                   nn.BatchNorm2d(self.out_channels, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        bsz = x.shape[0]
        x = self.trans(x)
        x = x.reshape(bsz, -1)
        
        return x

class Relational_Encoder(nn.Module):
    """
    Encodes relational information for points, normals, and colors.

    Args:
        M (int): Maximum kernel size for the second dimension.
        N (int): Maximum kernel size for the first dimension.

    Attributes:
        p_encoder (R_Encoder): Encoder for point features.
        n_encoder (R_Encoder): Encoder for normal features.
        c_encoder (R_Encoder): Encoder for color features.
        feat_dim (int): Combined feature dimension of all encoders.
    """
    def __init__(self, M, N):
        super(Relational_Encoder, self).__init__()
        self.p_encoder = R_Encoder(kernel_size=16, kernel_size2=2, max_kernel_size=N, max_kernel_size2=M)
        self.n_encoder = R_Encoder(kernel_size=16, kernel_size2=2, max_kernel_size=N, max_kernel_size2=M)
        self.c_encoder = R_Encoder(kernel_size=16, kernel_size2=2, max_kernel_size=N, max_kernel_size2=M)

        self.feat_dim = self.p_encoder.feat_dim + self.n_encoder.feat_dim + self.c_encoder.feat_dim

    def forward(self, x):
        points, normals, colors = x[:,:,:,:3].transpose(3,1), x[:,:,:,3:6].transpose(3,1), x[:,:,:,6:].transpose(3,1)
        x1 = self.p_encoder(points)
        x2 = self.n_encoder(normals)
        x3 = self.c_encoder(colors)
        x = torch.cat((x1,x2,x3), dim=-1)
        
        return x

class Global_Encoder(nn.Module):
    """
    Encodes global features using 1D convolutions.

    Args:
        input_dim (int): Dimensionality of the input.
        feat_dim (int, optional): Dimensionality of the output features. Default is 384.

    Attributes:
        feat_paths (nn.Sequential): First stage of convolutional layers.
        feat_all (nn.Sequential): Second stage of convolutional layers.
    """
    def __init__(self, input_dim, feat_dim=384):
        super(Global_Encoder, self).__init__()
        self.feat_dim=feat_dim

        self.feat_paths = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1),
                                   nn.BatchNorm1d(32, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
                                   nn.BatchNorm1d(64, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
                                   nn.BatchNorm1d(128, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                  )

        self.feat_all = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
                                   nn.BatchNorm1d(256, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
                                   nn.BatchNorm1d(256, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=256, out_channels=self.feat_dim, kernel_size=1),
                                   nn.BatchNorm1d(self.feat_dim, momentum=0.1),
                                   nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        shape = x.shape

        x = x.reshape((shape[0]*shape[1], shape[2], shape[3]))
        x = x.transpose(2,1)
        x = self.feat_paths(x)
        x = x.transpose(2,1)
        x = x.reshape((shape[0],shape[1], shape[2], -1))
        x = torch.max(x, dim=2)[0]

        x = x.transpose(2,1)
        x = self.feat_all(x)
        x = x.transpose(2,1)
        x = torch.max(x, dim=1)[0]
        
        return x

class Encoder(nn.Module):
    """
    Combines relational and global encoders into a single module.

    Args:
        input_dim (int): Dimensionality of the input.
        M (int): Number of paths.
        N (int): Number of vertices per path.

    Attributes:
        global_encoder (Global_Encoder): Global encoder module.
        relational_encoder (Relational_Encoder): Relational encoder module.
        feat_dim (int): Combined feature dimensionality.
    """
    def __init__(self, input_dim, M, N):
        super(Encoder, self).__init__()

        self.global_encoder = Global_Encoder(input_dim=input_dim)
        self.relational_encoder = Relational_Encoder(M, N)

        self.feat_dim = self.global_encoder.feat_dim + self.relational_encoder.feat_dim

    def forward(self, x):
        x1 = self.global_encoder(x[:,:,-8:,:])
        x2 = self.relational_encoder(x)
        x = torch.cat((x1, x2), dim=-1)

        return x

class Classifier(nn.Module):
    """
    A classification module built on fully connected layers.

    Args:
        feat_dim (int): Dimensionality of the input features.
        num_classes (int): Number of classes.

    Attributes:
        classifier (nn.Sequential): Fully connected layers for classification.
    """
    def __init__(self, feat_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(feat_dim, 256),
                                   nn.BatchNorm1d(256, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5),
                                   nn.Linear(256, num_classes)
                                  )

    def forward(self, x):
        return self.classifier(x)

class Model(nn.Module):
    """
    Complete model combining the encoder and classifier.

    Args:
        input_dim (int, optional): Dimensionality of the input. Default is 9.
        M (int, optional): Number of paths. Default is 16.
        N (int, optional): Number of vertices per path. Default is 16.
        num_classes (int, optional): Number of classes. Default is 6.

    Attributes:
        encoder (Encoder): Encoder module.
        classifier (Classifier): Classifier module.
    """
    def __init__(self, input_dim=9, M=16, N=16, num_classes=6):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, M=M, N=N)
        self.classifier = Classifier(feat_dim=self.encoder.feat_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)

        return x
