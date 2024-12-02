import torch
import torch.nn as nn
from torch import linalg

from mmdet.registry import MODELS
from .utils import weighted_loss
from mmdet.models.reid import GlobalAveragePooling
from mmengine.device import get_device


@weighted_loss
def cluster_loss(statsSum,target=None):
   assert statsSum['SS+']['N'] > 0 and statsSum['SS-']['N']>0, "SS+ "+str(statsSum['SS+']['N'])+" SS- "+str(statsSum['SS-']['N'])
   """
   lossP = (torch.norm(statsSum['SS+']['S'] - statsSum['SS+']['SS'],p=2) ** 2)/statsSum['SS+']['N']
   lossN = (torch.norm(statsSum['SS-']['S'] - statsSum['SS-']['SS'],p=2) ** 2)/statsSum['SS-']['N']
   """ 
   lossP = ((torch.pow(statsSum['SS+']['S'], 2) + \
      torch.pow(statsSum['SS+']['SS'], 2).unsqueeze(0).expand(statsSum['SS+']['N'], -1) - \
      2 * statsSum['SS+']['S'] * statsSum['SS+']['SS'].unsqueeze(0).expand(statsSum['SS+']['N'], -1)).sum(dim=1)).sum(dim=0)/ statsSum['SS+']['N']
   lossN = ((torch.pow(statsSum['SS-']['S'], 2) + \
      torch.pow(statsSum['SS-']['SS'], 2).unsqueeze(0).expand(statsSum['SS-']['N'], -1) - \
      2 * statsSum['SS-']['S'] * statsSum['SS-']['SS'].unsqueeze(0).expand(statsSum['SS-']['N'], -1)).sum(dim=1)).sum(dim=0)/ statsSum['SS-']['N']
   euclidean_distance = (torch.norm(statsSum['SS+']['SS'] - statsSum['SS-']['SS'],p=2) ** 2) 

   loss = (lossP+lossN)/(euclidean_distance+1e-8)
   return loss

@MODELS.register_module()
class ClusterLoss(nn.Module):

   def __init__(self, reduction='mean', loss_weight=1.0):
      super(ClusterLoss, self).__init__()
      self.reduction = reduction
      self.loss_weight = loss_weight
      self.gap = GlobalAveragePooling(kernel_size=7)
      self.centroides = nn.Parameter(torch.randn((2,256), device=get_device()), requires_grad=True)
                            
   def forward(self,
               pos_embbeds,
               neg_embbeds,
               weight=None,
               avg_factor=None,
               reduction_override=None):
      assert reduction_override in (None, 'none', 'mean', 'sum')
      reduction = (reduction_override if reduction_override else self.reduction)
      
      SSp={'S':None,'SL':None,'SS':None, "N":0}
      SSn={'S':None, 'SL':None,'SS':None, "N":0}
      statsSum = {'SS+':SSp,'SS-':SSn}
               
      pos_embbeds = self.gap(pos_embbeds)
      neg_embbeds = self.gap(neg_embbeds)
      SSp['S'] = pos_embbeds
      SSp['SL'] = torch.sum(pos_embbeds,dim=0)
      SSp['N'] = pos_embbeds.shape[0]

      SSn['S'] = neg_embbeds
      SSn['SL'] = torch.sum(neg_embbeds,dim=0)
      SSn['N'] = neg_embbeds.shape[0]

      SSp['SS'] = self.centroides[0]
      SSn['SS'] = self.centroides[1]
      loss_bbox = self.loss_weight * cluster_loss(statsSum,None)
      
      return loss_bbox
