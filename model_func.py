import torch
from torch import nn, optim
from preprocess_data import *
from transformer_neural_network import TransformerNetwork


# Training loop 
def train(dataloader, model, loss_fn, optimizer, train_epoch):
    model.train()
    num_batches = len(dataloader) -1
    train_loss, correct = 0, 0


    for batch, (x, y) in enumerate(dataloader):
        total_guess = 0
        x = x.squeeze(1).to(device)
        
        # label = torch.tensor(y, dtype=torch.float, requires_grad=False).to(device)
        label = y.clone().detach().to(device)
        prediction = model(x)

        loss = loss_fn(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(label.size(0)):
                correct += (label[i].argmax(1) == prediction[i].argmax(1)).type(torch.float).sum().item()
                total_guess += 6

        accuracy = correct/total_guess

        if train_epoch % 1 == 0 :
            print(f"Total Loss :  {loss.item():>7f} | Accuracy : {correct}/{total_guess} = {accuracy*100:>4f}%")
        

# Testing loop 
def evaluate(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    total_guess = 0
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:

            x = x.squeeze(1).to(device)
            label = torch.tensor(y, dtype=torch.float, requires_grad=False).to(device)
            prediction = model(x)
            test_loss += loss_fn(prediction, label).item()

            for i in range(label.size(0)):
                correct += (label[i].argmax(1) == prediction[i].argmax(1)).type(torch.float).sum().item()
                total_guess += 6

        test_loss /= num_batches
        accuracy = correct/total_guess

        print(f"Test_loss : {test_loss:>7f}   |   Accuracy : {correct}/{total_guess} = {accuracy*100:>4f}%\n")


# Predict the answer
def predict(dataloader, model):

    with torch.no_grad():
        for x, y in dataloader:
            x = x.squeeze(1).to(device)
            prediction = model(x)

            for i in range(prediction.size(0)):
                pred = prediction[i].argmax(1)

                answer = []
                for value in pred:
                    value = value.item()
                    if value == 0:
                        answer.append(0.0)
                    elif value == 1:
                        answer.append(0.5)
                    elif value == 2:
                        answer.append(1.0)
                    elif value == 3:
                        answer.append(1.5)
                    elif value == 4:
                        answer.append(2.0)
                    elif value == 5:
                        answer.append(2.5)
                    elif value == 6:
                        answer.append(3.0)
                    elif value == 7:
                        answer.append(3.5)
                    elif value == 8:
                        answer.append(4.0)
                    elif value == 9:
                        answer.append(4.5)
                    elif value == 10:
                        answer.append(5.0)
                    
                print("Answer ", i, " : ", answer)


# This function used to test and debug purpose
def demonstrate_train():
    data = preprocess_data("Feedback Prize - English Language Learning\\data\\train.csv", True)
    datasets = CustomStrokeDataset(data)
    train_dataloader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = "cuda:0"if torch.cuda.is_available else "cpu"

    learning_rate = 1.0

    model = TransformerNetwork().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # evaluate(train_dataloader, model, criterion)
    predict(train_dataloader, model)


# demonstrate_train()
