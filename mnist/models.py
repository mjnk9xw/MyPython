from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(14 * 14 * 32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out