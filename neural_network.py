import torch
import torch.nn as nn

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.001)

''' used until 22th August
class NeuralNetwork(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.features = features

        layers = []

        for i in range(len(features) - 1):
            
            linear_layer = nn.Linear(features[i], features[i + 1])
            layers.append(linear_layer)

            if i < len(features) - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.model(torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=-1))
'''

''' ResNet Antipattern
class NeuralNetwork(nn.Module):
    def __init__(self, features, use_res: bool = False):
        super().__init__()
        self.features = features
        self.use_res = use_res

        self.layers = nn.ModuleList([
            nn.Linear(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])
        self.act = nn.Tanh()

    def forward(self, x, t):
        h = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=-1)

        for i, layer in enumerate(self.layers):
            out = layer(h)
            
            if i < len(self.layers) - 1:
                out = self.act(out)
                
                if self.use_res and h.shape[-1] == out.shape[-1]:
                    out = out + h

            h = out

        return h
'''

class NeuralNetwork(nn.Module):
    def __init__(self, features, use_res: bool = False):
        super().__init__()
        self.features = features
        self.use_res = use_res

        self.layers = nn.ModuleList([
            nn.Linear(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])
        self.act = nn.Tanh()

    def forward(self, x, t):
        h = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=-1)

        res = None
        skip_count = 0

        for i, layer in enumerate(self.layers):
            h = layer(h)

            # Nur Hidden-Layer erhalten Aktivierung und Residuals
            if i < len(self.layers) - 1:
                h = self.act(h)

                if self.use_res:
                    # Ankerpunkt setzen (z. B. nach dem Input-Layer oder bei Dimensionswechsel)
                    if res is None or res.shape[-1] != h.shape[-1]:
                        res = h
                        skip_count = 0
                    else:
                        skip_count += 1
                        # Nach 2 Schichten aufaddieren und neuen Anker setzen
                        if skip_count == 2:
                            h = h + res
                            res = h
                            skip_count = 0

        return h

# for toy problem
class ODENeuralNetwork(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.features = features

        layers = []

        for i in range(len(features) - 1):
            
            linear_layer = nn.Linear(features[i], features[i + 1])
            layers.append(linear_layer)

            if i < len(features) - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, t):
        return self.model(t)