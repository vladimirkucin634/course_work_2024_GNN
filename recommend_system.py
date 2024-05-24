from copy import deepcopy
from sentence_transformers import SentenceTransformer
from dadata import Dadata
from const import DADATA_TOKEN, DADATA_SECRET
from modeling import preproc_strigs, EdgePred, DotPredictor
import pandas as pd
import dgl
import numpy as np
import torch
from MODEL import MODEL
sent_transf_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
okved = pd.read_csv('okved.csv', index_col='Код')
tins = pd.read_csv('tin_legal.csv', index_col=0)
dadata = Dadata(DADATA_TOKEN, DADATA_SECRET)
g = dgl.load_graphs('graph2.bin')[0][0]
mean = np.load('mean.npy')
sigma = np.load('sigma.npy')
model = torch.load('model.pth')
def recommend_system(tin, price, supplier=True, contract='', topk=5):

    company_info = dadata.find_by_id(name='party', query=str(tin))[0]['data']
    city_lat = company_info['address']['data']['geo_lat']
    city_lng = company_info['address']['data']['geo_lon']
    okved_code = company_info['okved']
    
    geo_feat = np.array([city_lat, 
                          city_lng, 
                          price, 
                          price, 
                          price,
                          price,
                          1 if supplier else 0, 
                          0 if supplier else 1]).astype(np.float32)
    
    act_feat = sent_transf_model.encode([preproc_strigs(okved.loc[okved_code].values[0])])[0]
    contact_feat = sent_transf_model.encode([preproc_strigs(contract)])[0]
    geo_feat = (geo_feat - mean) / sigma
    geo_feat = geo_feat.astype(np.float32)
    geo_feat = torch.from_numpy(geo_feat)
    geo_feat = geo_feat.unsqueeze(0)
    geo, act, weight = g.ndata['feat'], g.ndata['activity'], g.edata['weight']
    g_ = deepcopy(g)
    g_.add_nodes(1)
    g_.ndata['feat'] = torch.cat([geo, torch.Tensor(geo_feat).reshape(1, -1)])
    g_.ndata['activity'] = torch.cat([act, torch.Tensor(act_feat).reshape(1, -1)])   

    new_edges = []

    if supplier:
        for i in range(g_.number_of_nodes() - 1):
            new_edges.append((i, g_.number_of_nodes() - 1))
    else:
        for i in range(g_.number_of_nodes() - 1):
            new_edges.append((g_.number_of_nodes() - 1, i))
        
    new_edges = np.array(new_edges)
    g_.add_edges(new_edges[:, 0], new_edges[:, 1])
    g_.edata['weight'] = torch.cat([weight, torch.Tensor(contact_feat).repeat(new_edges.shape[0], 1)])

    theta = 0.9
    pred_act = EdgePred()
    pred_geo = DotPredictor()
    with torch.no_grad():
        h1, h2 = model(g_, g_.ndata['feat'], g_.ndata['activity'], g_.edata['weight'], features='sep')        
        pos_score_act = pred_act(g_, h1)
        pos_score_geo = pred_geo(g_, h2)
        pos_score = theta * pos_score_act + (1 - theta) * pos_score_geo
        pos_score = pos_score.detach().numpy()

    topk = np.argsort(pos_score)[-topk:]
    topk = topk[::-1]
    edges = g_.edges()
    if supplier:
        res_tins = tins.loc[np.array(edges[0])[[*topk]]]['ИНН'].values
    else:
        res_tins = tins.loc[np.array(edges[1])[[*topk]]]['ИНН'].values
    
    for tin in res_tins:
        data_tin = dadata.find_by_id(name="party", query=str(tin))[0]["data"]
        print(f'ИНН: {tin}, название: {data_tin["name"]["short_with_opf"]}, юр. адрес: {data_tin["address"]["value"]}')
