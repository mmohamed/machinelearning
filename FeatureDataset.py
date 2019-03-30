from __future__ import division
import torch
import time
import copy


class FeatureDataset():
    def __init__(self, X_train, Y_train, transform=None):
    
        self.X_train = X_train
        self.Y_train = Y_train
        self.transform = transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return X_train[idx].astype(float), Y_train[idx].astype(float)

def trainModel(model, criterion, optimizer, numEpochs, dsetLoaders, dsetSizes):
    
    since = time.time()

    bestModel = model
    bestACC = 0.0
    useGPU = 1
    
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                mode = 'train'
                #optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
                print("TRAINING STARTED")
            else:
                model.eval()
                mode = 'val'
                print("TESTING STARTED")

            runningLoss = 0.0
            runningCorrects = 0

            counter = 0
            # Iterate over data.
            answer = []
            for data in dsetLoaders[phase]:
                inputs, labels = data 
                
                # wrap them in Variable
                if useGPU:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()),                             
                        Variable(labels.long().cuda())
                    except:
                        print(inputs,labels)
                else:
                    inputs, labels = Variable(Variable(inputs).float()), Variable(Variable(labels).float())

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                for i in range(len(outputs.data)):
                    answer.append(outputs.data[i])
                _, preds = torch.max(outputs.data, 1)
                #print(outputs.data
                #print(_, preds
                loss = criterion(outputs, labels)
                print('loss done')                
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.
                if counter%50 == 0:
                    print("Reached iteration ",counter)
                counter += 1

                # backward + optimize only if in training phase
                if phase == 'train':
                    print('loss backward')
                    loss.backward()
                    print('done loss backward')
                    optimizer.step()
                    print('done optim')
                # print(evaluation statistics
                try:
                    runningLoss += loss.data[0]

                    print(preds.shape, labels.data.shape)
                    for q in range(len(labels.data)):
                        if labels.data[q][preds[q]] == 1:
                            runningCorrects += 1
                    
                    #runningCorrects += torch.sum(preds == labels.data.long())
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epochLoss = runningLoss / dsetSizes[phase]
            epochACC = runningCorrects / dsetSizes[phase]
            print(phase, runningCorrects, dsetSizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochACC))

            # deep copy the model
            if phase == 'test':
                if epochACC > bestACC:
                    bestACC = epochACC
                    bestModel = copy.deepcopy(model)
                    print('new best accuracy = ',bestACC)

    timeElapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
    print('Best val Acc: {:4f}'.format(bestACC))
    print('returning and looping back')
    #torch.save(bestModel.state_dict(), 'fine_tuned_bestModel.pt')
    return bestModel, answer