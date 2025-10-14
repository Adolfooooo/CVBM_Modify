import torch
import torch.nn as nn

class PseudoLabelOptimizer(nn.Module):
    def __init__(self, threshold_low=0.3, threshold_high=0.7):
        super(PseudoLabelOptimizer, self).__init__()
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def forward(self, pseudo_labels):
        confident = pseudo_labels > self.threshold_high
        hesitant = (pseudo_labels <= self.threshold_high) & (pseudo_labels >= self.threshold_low)
        scrap = pseudo_labels < self.threshold_low
        
        return confident, hesitant, scrap
    
    def calculate_loss(self, predictions, labels, confident_mask, hesitant_mask):
        loss_confident, loss_hesitant= 0, 0
        if confident_mask.sum() > 0:
            loss_confident = nn.functional.mse_loss(predictions[confident_mask].float(), labels[confident_mask].float())
        if hesitant_mask.sum() > 0:
            loss_hesitant = nn.functional.smooth_l1_loss(predictions[hesitant_mask].float(), labels[hesitant_mask].float())
        
        return loss_confident + loss_hesitant
