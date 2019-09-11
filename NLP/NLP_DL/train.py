import torch
from torch.utils import data
import torch.nn as nn
import tqdm
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()
    

class ListDataset(data.Dataset):
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(a_list) for a_list in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(a_list[index] for a_list in self.lists)

    def __len__(self):
        return len(self.lists[0])

def pad_seq(batches):
    batches.sort(key=lambda x: len(x[0]), reverse=True)
    features, targets = zip(*batches)
    padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    return padded, torch.as_tensor(targets)

class MeanMax(nn.Module):
    def __init__(self):
        super(MeanMax,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((None,1))
        self.max = nn.AdaptiveMaxPool2d((None,1))
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return self.alpha*self.mean(x)+(1-self.alpha)*self.max(x)
    
class Attention(nn.Module):
    def __init__(self,input_dim): # B* T * D
        super(Attention, self).__init__()
        self.linear=nn.Linear(input_dim,1) # B * T * 1
        
    def forward(self, x): # B * T * D
        mean_D = self.linear(x) # B * T * 1
        weight = torch.softmax(mean_D,dim=1) 
        outcome = weight * x
        return outcome, weight
              
class Cnn(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Cnn,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),            
            MeanMax(),            
        )
        self.attention = Attention(64)
        self.output_layer = nn.Linear(64,output_dim)
        
    def forward(self, X):
        X= X.unsqueeze(1) # B x 1 x T x D
        y = self.model(X) # B x C x T x 1
        y = y.squeeze(-1) # B x C x T
        y = y.reshape(y.shape[0], y.shape[2],y.shape[1])# B x T x C
        after_attention, weight = self.attention(y) # B x T x C
        return  self.output_layer(after_attention.mean(1)), weight # self.output_layer(y.mean(1)),_

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # load data
    default_path = os.environ['SM_INPUT_DIR']
    pk = open(os.path.join(default_path, "data/training/train_X.pkl"), "rb") 
    train_X = pickle.load(pk)
    pk = open(os.path.join(default_path, "data/training/train_y.pkl"), "rb")
    train_y = pickle.load(pk)
    
    train_X = [torch.as_tensor(a) for a in train_X]
    dataset = ListDataset(train_X, train_y)
    dataloader = data.DataLoader(dataset, batch_size=args.batchsize, collate_fn=pad_seq,shuffle=True)
    my_cnn = Cnn(None,len(train_y)).to(device)
        
    optimizer = torch.optim.Adam(my_cnn.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    with torch.set_grad_enabled(True):
        with tqdm.tqdm(total=args.epochs, unit='epoch',leave=True) as t: # not working
            for epoch in range(args.epochs):
                print('epoch: ', epoch)
                epochloss = 0
                epochaccuracy = 0
                for input_x, target in dataloader:
                    optimizer.zero_grad()
                    output,_ = my_cnn(input_x) 

                    accuracy = (output.argmax(dim=1)==target).sum().float()/len(target)*100
                    epochaccuracy += accuracy
                    loss = criterion(output, target.long())
                    loss.backward()
                    optimizer.step() 
                    epochloss += loss.item()
                t.set_postfix(loss=epochloss/len(dataloader),accuracy = epochaccuracy/len(dataloader)) 
                print('epochaccuracy: ',epochaccuracy/len(dataloader))
                print('epochloss: ',epochloss/len(dataloader))
                t.update()
                
    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model.pth') 
    torch.save(my_cnn, model_path)
    print('Finished.')