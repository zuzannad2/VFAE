import torch
from torch.nn import Module, Linear, ReLU, Dropout, Sigmoid

class MLP(Module):
    '''
    Single layer MLP
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, activation) -> None:
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim) 
        self.activation = activation
        self.fc2 = Linear(hidden_dim, latent_dim)
        
    def forward(self, inputs):
        output = self.activation(self.fc1(inputs))
        return self.fc2(output)

class VariationalMLP(Module):
    '''
    Single layer MLP with sampling a latent z
    '''
    def __init__(self, input_dim, hidden_dim, z_dim, activation) -> None:
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.activation = activation
        self.fc_logvar = Linear(hidden_dim, z_dim)
        self.fc_mean = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        logvar = self.fc_logvar(outputs).mul(.5).exp()
        mean = self.fc_mean(outputs)
        eps = torch.randn_like(mean) 
        z = eps * logvar + mean
        
        return z, logvar, mean

class VFAE(Module):
    '''
    Variational Fair Autoencoder model implementation
    '''
    def __init__(
        self,
        dim_x,
        dim_s,
        dim_y, 
        dim_z1_enc, 
        dim_z2_enc, 
        dim_z1_dec, 
        dim_x_dec, 
        dim_z,
        dropout) -> None:
        
        super().__init__()
        self.activation = Sigmoid()
        dim_y_tmp = 2 if dim_y == 1 else dim_y

        self.encoder_z1 = VariationalMLP(dim_x+dim_s, dim_z1_enc, dim_z, self.activation)
        self.encoder_z2 = VariationalMLP(dim_z+dim_y, dim_z2_enc, dim_z, self.activation)

        self.decoder_z1 = VariationalMLP(dim_z+dim_y, dim_z1_dec, dim_z, self.activation)
        self.decoder_y = MLP(dim_z, dim_x_dec, dim_y_tmp, self.activation)
        self.decoder_x = MLP(dim_z+dim_s, dim_x_dec, dim_x+dim_s, self.activation)

        self.dropout = Dropout(dropout)

    def forward(self, inputs):
        x, y, s = inputs['x'], inputs['y'], inputs['s'] 
        # Encode
        x_s = torch.cat([x,s], dim=1)
        x_s = self.dropout(x_s)

        z1_enc, z1_enc_logvar, z1_enc_mean = self.encoder_z1(x_s)
        z1_y = torch.cat([z1_enc,y], dim=1)
        z2_enc, z2_enc_logvar, z2_enc_mean = self.encoder_z2(z1_y)

        # Decode
        z2_y = torch.cat([z2_enc, y], dim=1)
        z1_dec, z1_dec_logvar, z1_dec_mean = self.decoder_z1(z2_y)

        z1_s = torch.cat([z1_dec, s], dim=1)
        x_dec = self.decoder_x(z1_s)

        y_dec = self.decoder_y(z1_enc)
        
        out = {
            'x_pred' : x_dec, 
            'y_pred' : y_dec, 
            'z1_enc' : z1_enc,

            'z1_enc_logvar': z1_enc_logvar, 
            'z1_enc_mean': z1_enc_mean, 
            
            'z2_enc_logvar': z2_enc_logvar,
            'z2_enc_mean': z2_enc_mean,
            
            'z1_dec_logvar': z1_dec_logvar,
            'z1_dec_mean': z1_dec_mean
        }
        return out


def train(epoch, model, train_dataloader, loss_function, optimizer, print_freq = 50):
    #model.train()
    running_loss = 0
    count = 0
    
    for batch_id, data in enumerate(train_dataloader):
        preds_dict = model(data)
        loss = loss_function(data, preds_dict)
        loss.backward()
        optimizer.step()

        running_loss += loss
        count += data['x'].size(0)

        if batch_id % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * data['x'].size(0), len(train_dataloader.dataset), 100. * batch_id / len(train_dataloader), loss))
    
    running_loss /= count
    print('Train Epoch: {} Average loss: {:.4f}'.format(epoch, running_loss))

    return running_loss
