import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_classes=37, init_weights=True):

        self.kernel_size = 3

        super(CNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.kernel_size)

        # Define the fully connected layers
        # self.fc1 = nn.Linear(15*1024, 4800)
        self.fc1 = nn.Linear(15872, 4800)  # 15360, 14848, 14336, 14080
        self.fc2 = nn.Linear(4800, 3200)
        self.fc3 = nn.Linear(3200, 1600)
        self.fc4 = nn.Linear(1600, n_classes)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # Define activation functions and pooling layer
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout()

        # Define the output activation function (e.g., sigmoid for binary classification)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the tensor before passing it through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)

        # x = self.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
if __name__ == "__main__":
    from time import time
    from thop import profile
    input = torch.randn((16,1,1024))
    net = CNN()
    t1 = time()
    output = net(input)
    t2=time()
    print((t2-t1)/16*1000)
    flops, params = profile(net, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')