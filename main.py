from preprocess_data import *
from transformer_neural_network import TransformerNetwork
from torch import nn, optim
from model_func import *

# Set up GPU Environment
devive = "cuda:0"if torch.cuda.is_available else "cpu"

# --------- Build Model -------------#

model = TransformerNetwork().to(device)

# --------- Prepare Training Data -------------#

processed_data = preprocess_data("Feedback Prize - English Language Learning\\data\\train.csv", True)
train_set, test_set = split_dataset(processed_data)

train_dataset = CustomStrokeDataset(train_set)
test_dataset = CustomStrokeDataset(test_set)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# --------- Hyper parameter setting -------------#

learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10000

# --------- Train and Test -------------#


for epoch in range(epochs):
    print()
    print("="*30) 
    print("Epoch ",epoch)  
    train(train_dataloader, model, criterion, optimizer, epoch)
    evaluate(test_dataloader, model, criterion)


print("Done!")

# --------- Predict The Result -------------#

def predict_answer():
    data = preprocess_data("Feedback Prize - English Language Learning\\data\\test.csv", True)
    predict_dataset = CustomStrokeDataset(data)
    predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=True)

    predict(predict_dataloader, model)


def save_model():
    while True:
        save_response = input("Save the model? [y/n]")

        if save_response == 'y' or save_response == 'yes':
            torch.save(model.state_dict(), 'eng_score_predict_model_weights.pth')
        elif save_response == 'n' or save_response == 'no':
            break
        else:
            print("Please enter yes/no or y/n")


predict_answer()
# save_model()