import torch 
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

class Fast_MMD():
    
    def __init__(self, gamma, features_out) -> None:
        self.gamma = gamma
        self.features_out = features_out

    def forward(self, z1, z2):
        features_in = z1.shape[-1]
        
        w = torch.randn((features_in, self.features_out))
        b = torch.zeros((self.features_out,)).uniform_(0,2 * np.pi)
        
        psi_z1 = self.psi(z1, w, b).mean(dim=0)
        psi_z2 = self.psi(z2, w, b).mean(dim=0)

        return torch.norm(psi_z1 - psi_z2, 2)

    def psi(self, x, w, b):
        return np.sqrt(2 / self.features_out) * (np.sqrt(2 / self.gamma) * (x @ w + b)).cos()

class VFAE_loss():

    def __init__(self, alpha, beta, gamma, dims_out) -> None:
        self.alpha = alpha 
        self.beta = beta

        self.ce_loss = CrossEntropyLoss()
        self.bce_loss = BCEWithLogitsLoss()
        self.mmd = Fast_MMD(gamma, dims_out)

    def forward(self, y_true, y_pred):
        x, s, y = y_true['x'], y_true['s'], y_true['y']
        x_s = torch.cat([x, s], dim=-1)

        supervised_loss = self.ce_loss(y_pred['y_pred'], y)
        reconstruction_loss = self.bce_loss(y_pred['x_pred'], x_s)

        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss_z1 = self.kl_divergence(y_pred['z1_enc_logvar'], y_pred['z1_dec_logvar'],  y_pred['z1_enc_mu'], y_pred['z1_dec_mu'])
        kl_loss_z2 = self.kl_divergence(y_pred['z2_enc_logvar'], y_pred['z2_enc_mu'], zeros, zeros)
        
        vfae_loss = supervised_loss + kl_loss_z1 + kl_loss_z2 + self.alpha * reconstruction_loss

        z1_encoded = y_pred['z1_enc']
        z1_sensitive, z1_nonsensitive = self.separate_sensitive(z1_encoded, s)
        vfae_loss += self.beta * self.mmd(z1_sensitive, z1_nonsensitive)
        
        return vfae_loss


    @staticmethod
    def kl_divergence(logvar_z1, logvar_z2, mu_z1, mu_z2):
        kl = 0.5 * torch.sum(logvar_z1 - logvar_z2 + (logvar_z1.exp() + (mu_z1 - mu_z2)**2) / logvar_z2.exp(), dim=1)
        return kl.mean()
    
    @staticmethod
    def separate_sensitive(variables, s):
        sensitive_ix = (s == 1).nonzero()[:, 0]
        nonsensitive_ix = (s == 0).nonzero()[:, 0]

        sensitive = variables[sensitive_ix]
        nonsensitive = variables[nonsensitive_ix]

        return sensitive, nonsensitive



    
                    
