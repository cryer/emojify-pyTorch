import torch
import os
from emo_utils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = int(word_to_index[w])
            j = j + 1
    return X_indices


class emojiDataset(Data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if download:
            pass
        if self.train:
            traindata,trainlabels = read_csv(os.path.join(self.root,'mytrain.csv'))
            self.train_data = sentences_to_indices(traindata, word_to_index, 10)
            self.train_labels = trainlabels
            # self.train_labels = convert_to_one_hot(trainlabels, C=5)
        else:
            pass
    def __getitem__(self, index):
        if self.train:
            data, target = self.train_data[index], self.train_labels[index]
        else:
            pass
        if self.transform is not None:
            pass
        if self.target_transform is not None:
            pass
        return data, target
    def __len__(self):
        if self.train:
            return 180
        else:
            return 0

train_data = emojiDataset(
    root='./data/',
    train=True,
    # transform=torchvision.transforms.ToTensor(),
    download=False,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=20, shuffle=True, num_workers=0)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1  #word index begin with 1,plus 1 for padding 0
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    return emb_matrix


class myModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,pretrained_weight):
        super(myModel,self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(pretrained_weight)
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.LSTM(embedding_dim, 128, 2,batch_first=True,dropout=0.5)
        self.linear = nn.Linear(128,5)
        self.out = nn.Softmax()

    def forward(self,x,h):
        out = self.word_embeds(x)
        out, _ = self.rnn(out,h)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.out(out)
        return out

vocab_len = len(word_to_index) + 1
emb_matrix = pretrained_embedding_layer(word_to_vec_map, word_to_index)
model = myModel(vocab_len,50,emb_matrix)
# model.word_embeds.parameters().requires_grad  = False

for param in model.parameters():
    param.requires_grad = False
    break


model = model.cuda()

loss_func = nn.CrossEntropyLoss()

optimizer1 = torch.optim.Adam(model.rnn.parameters(),lr=0.001)
optimizer2 = torch.optim.Adam(model.linear.parameters(),lr=0.001)


for epoch in range(50):
    for step,(data,target) in enumerate(train_loader):
        data = data.long()
        target = target.long()
        model.zero_grad()
        states = (Variable(torch.zeros(2, 20, 128)).cuda(), Variable(torch.zeros(2, 20, 128)).cuda())
        input = Variable(data).cuda()
        target = Variable(target).cuda()
        output = model(input,states)
        loss = loss_func(output,target)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    if (epoch+1) % 10 ==0:
        x_test = np.array(['not feeling happy', 'Holy shit', 'you are so pretty', 'let us play ball'])
        X_test_indices = sentences_to_indices(x_test, word_to_index, 10)
        X_test_indices = torch.from_numpy(X_test_indices)
        X_test_indices = Variable(X_test_indices.long()).cuda()
        states = (Variable(torch.zeros(2, 4, 128)).cuda(), Variable(torch.zeros(2, 4, 128)).cuda())
        pred = model(X_test_indices,states)
        for i in range(len(x_test)):
            num = np.argmax(pred.data[i])
            print(' prediction: ' + x_test[i] + label_to_emoji(num).strip())

torch.save(model.state_dict(),"emojify.pkl")

