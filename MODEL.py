from torch import nn
import torch
from dgl.nn import GINEConv, SAGEConv, GINConv
import torch.nn.functional as F
class MODEL(nn.Module):
    def __init__(self, 
                 geo_layer,
                 in_feats_geo, 
                 h_feats_geo, 
                 in_feats_act, 
                 h_feats_act,
                 num_geo_layers=2,
                 num_act_layers=2,
                 activation=F.leaky_relu,
                 dropout_geo=0.1,
                 dropout_act=0.1):
        super(MODEL, self).__init__()
        
        self.activation = activation

        # action features part
        self.act_layers = nn.ModuleList()
        for _ in range(num_act_layers):
            self.act_layers.append(GINEConv(nn.Linear(in_feats_act, h_feats_act), learn_eps=False))

        self.fc = nn.Linear(h_feats_act, in_feats_act)
        self.drop_act = nn.Dropout(p=dropout_act)
        
        # geo and price features part
        self.geo_layers = nn.ModuleList()
        if geo_layer == GINConv:
            self.geo_layers.append(GINConv(nn.Linear(in_feats_geo, h_feats_geo)))
            for _ in range(num_geo_layers - 1):
                self.geo_layers.append(GINConv(nn.Linear(h_feats_geo, h_feats_geo)))
        elif geo_layer == SAGEConv:
            self.geo_layers.append(SAGEConv(in_feats_geo, h_feats_geo, aggregator_type='lstm'))
            for _ in range(num_geo_layers - 1):
                self.geo_layers.append(SAGEConv(h_feats_geo, h_feats_geo, aggregator_type='lstm'))
        
        self.drop_geo = nn.Dropout(p=dropout_geo)
    
    def forward(self, g, n_feat_geo, nfeat_act, efeat_act, features='both'):
        assert features in ['both', 'geo', 'act', 'sep'], \
            'features must be either "both", "geo", "sep" or "act"'
        
        h2 = n_feat_geo
        for layer in self.geo_layers:
            h2 = layer(g, h2)
            h2 = self.activation(h2)
            h2 = self.drop_geo(h2)
        
        h1 = self.act_layers[0](g, nfeat_act, efeat_act)
        h1 = self.fc(h1)
        for layer in self.act_layers[1:-1]:
            h1 = layer(g, h1, efeat_act)
            h1 = self.activation(h1)
            h1 = self.fc(h1)
            h1 = self.drop_act(h1)
        h1 = self.act_layers[-1](g, h1, efeat_act)
        
        return {
            'both': torch.cat([h1, h2], axis=1),
            'geo': h2,
            'act': h1,
            'sep': (h1, h2),
        }[features]
