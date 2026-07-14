import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes=3, input_channels=1, input_time=200, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.F2 = F2
        self.input_channels = input_channels
        self.input_time = input_time

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, F1, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, D * F1, kernel_size=(62, 1), groups=F1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, kernel_size=(1, 16), groups=D * F1),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        self._compute_flatten()

        self.classifier = nn.Linear(self.flatten_size, n_classes)
        self.emotion_head = self.classifier

    def _compute_flatten(self):
        dummy = torch.zeros(1, self.input_channels, 62, self.input_time)
        x = self.block1(dummy)
        x = self.depthwise(x)
        x = self.separable(x)
        self.flatten_size = x.view(1, -1).size(1)

    def forward(self, x, return_features=False, **kwargs):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        features = x.view(x.size(0), -1)
        out = self.classifier(features)
        if return_features:
            return out, features
        return out
