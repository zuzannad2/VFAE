import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class Test():
    def __init__(self, model, ckpt_path) -> None:
        '''
        :params:
            :checkpoint_path: Path of model checkpoint 
            :model: An instantiated model to load the checkpoint params to
        '''
        self.checkpoint_path = ckpt_path
        self.model = model
        self.model.load_state_dict(torch.load(ckpt_path, map_location = torch.device('cpu')))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()

    def __call__(self, dataloader):
        true = np.empty(len(dataloader))
        preds = np.empty(len(dataloader))
        with torch.no_grad():
            for data in dataloader:
                output = self.model(data)
                pred = torch.nn.functional.softmax(output['y_pred'], dim=1)
                pred = pred.argmax(1, keepdim=True)
                np.append(preds, pred)
                np.append(true, data['y'].item())
        return self.metrics(true, preds)
    
    @staticmethod
    def metrics(true, pred):
        accuracy = accuracy_score(true, pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(true, pred, average='weighted') 
        # TODO: Change this to a suitable one
        return accuracy, precision, recall, fscore

