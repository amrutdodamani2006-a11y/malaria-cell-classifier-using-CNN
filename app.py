class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)

def build_fastai_style_model():
    resnet = models.resnet34(weights=None)
    body = nn.Sequential(*list(resnet.children())[:-2])

    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASS_NAMES))
    )

    model = nn.Sequential(body, head)
    return model