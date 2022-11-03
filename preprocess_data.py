import matplotlib.pyplot as plt
import torch
import csv
import time
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from bert_utils import *


torch_tokenizer = get_tokenizer("basic_english")
device = "cuda:0"if torch.cuda.is_available else "cpu"


# Read data from csv and preprocess the data
def preprocess_data(path, score_bool):
    start = time.time()
    with open(path, 'r') as file_read:
        csv_reader = csv.reader(file_read)
        dataset = []
        k = 0
        # Loop through row
        for i, row in enumerate(csv_reader):
            
            row_data = []
            text_row = []
            score_row = []
            # Skip first row
            if i == 0:
                continue

            # Loop through column
            for j, column in enumerate(row):
                # Skip first row
                if j == 0:
                    continue

                # Process the long text data
                elif j == 1:
                    text = process_long_text(column)
                    text_row.append(text)

                # The process score data
                else:
                    score_row.append(column)
                
                    
            row_data.append(text_row)

            if score_bool:
                score_label = process_score_into_token(score_row)
                row_data.append(score_label)

            dataset.append(row_data)

            # Print progress of loading data
            k = k + 1
            if k % 100 == 0:
                print(k)

            # Control the row loop flow
            # if i == 10:
            #    break
        
        end = time.time()
        print("time : ", (end - start)/60, "minutes")
        print("The dataset is ready")
        # print(dataset)
        return dataset


# Split the dataset into train and test set based on ratio
def split_dataset(dataset, train_set_ratio=0.7):
    length_of_dataset = len(dataset)
    train_set = []

    # Loop until the i = ratio * length_of_dataset
    i = 0
    while i < int(length_of_dataset * train_set_ratio):
        random_index = torch.randint(len(dataset), size=(1,)).item()
        train_set.append(dataset[random_index])
        dataset.pop(random_index)
        i = i + 1

    test_set = dataset
    return train_set, test_set  


# Process the long text data 
# Split long string into small string with capacity of 200
def process_long_text(text):
    
    token_list = torch_tokenizer(text)
    sentence_length = len(token_list)
    sentences = []

    if sentence_length <= 200: 
        sentences.append(' '.join(token_list[0:sentence_length]))

    elif sentence_length <= 400:  
        sentences.append(' '.join(token_list[0:200]))   
        sentences.append(' '.join(token_list[200:sentence_length]))
    
    elif sentence_length <= 600:
        sentences.append(' '.join(token_list[0:200]))   
        sentences.append(' '.join(token_list[200:400]))      
        sentences.append(' '.join(token_list[400:sentence_length]))

    elif sentence_length <= 800:  
        sentences.append(' '.join(token_list[0:200]))   
        sentences.append(' '.join(token_list[200:400]))   
        sentences.append(' '.join(token_list[400:600])) 
        sentences.append(' '.join(token_list[600:sentence_length]))

    elif sentence_length <= 1000:   
        sentences.append(' '.join(token_list[0:200]))   
        sentences.append(' '.join(token_list[200:400]))   
        sentences.append(' '.join(token_list[400:600])) 
        sentences.append(' '.join(token_list[600:800]))
        sentences.append(' '.join(token_list[800:sentence_length]))

    else:
        sentences.append(' '.join(token_list[0:200]))   
        sentences.append(' '.join(token_list[200:400]))   
        sentences.append(' '.join(token_list[400:600])) 
        sentences.append(' '.join(token_list[600:800]))
        sentences.append(' '.join(token_list[800:1000]))
        sentences.append(' '.join(token_list[1000:sentence_length]))


    sentence_bert_values = []
    i = 0
    number_of_sentences = len(sentences)
    while i <= 5:
        if i <= number_of_sentences - 1:
            value = bert_function(sentences[i])
            
            sentence_bert_values.append(value)
        else:
            sentence_bert_values.append([0 for i in range(1024)])
        i = i + 1
    
    return sentence_bert_values

# Process the score values into tokens
# Score value range from 0 to 5 with increment of 0.5
def process_score_into_token(score_row):
    
    score_labels = []
    for value in score_row:
        value = float(value)

        if value == 0:
            score = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif value == 0.5:
            score = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif value == 1.0:
            score = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif value == 1.5:
            score = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif value == 2.0:
            score = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif value == 2.5:
            score = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif value == 3.0:
            score = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif value == 3.5:
            score = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif value == 4.0:
            score = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif value == 4.5:
            score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif value == 5.0:
            score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            score = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        score_labels.append(score)
    
    return score_labels


# Analyze the length of sentences and plot graph
def analyze_sentences_length():

    with open("Feedback Prize - English Language Learning\\data\\train.csv", 'r') as file_read:
        csv_reader = csv.reader(file_read)

        dataset = []
        for i, data in enumerate(csv_reader):
            if i == 0: continue

            dataset.append(data[1:])

    length_list = []
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    for data in dataset:
    
        sentence_length = len(torch_tokenizer(data[0]))
        length_list.append(sentence_length)
        
        
        if sentence_length < 200:
            a = a + 1
        elif sentence_length < 400:
            b = b + 1
        elif sentence_length < 600:
            c = c + 1
        elif sentence_length < 800:
            d = d + 1
        elif sentence_length < 1000:
            e = e + 1
        else:
            f = f + 1

    x_values = ["~200","~400","~600","~800","~1000","1k~"]
    y_values = [a, b, c, d, e, f]

    print(x_values)
    print(y_values)

    # Print list sort by sentence length, determine the longest sentence length
    # print(sorted(length_list))

    plt.bar(x_values, y_values)
    plt.xlabel("sentence length")
    plt.ylabel("Number")
    plt.show()
    

# Create a custom dataset which can be iterable by dataloader
class CustomStrokeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.dataset[idx][0]
        label = self.dataset[idx][1]              
        return torch.tensor(features).to(device), torch.tensor(label, dtype=torch.float).to(device)


# This function used to test and debug purpose
def demonstrate():

    data = preprocess_data("Feedback Prize - English Language Learning\\data\\train.csv")
    data1, data2 = split_dataset(data)
    # print(len(data1))
    # print(len(data2))
    datasets = CustomStrokeDataset(data1)
    train_dataloader = DataLoader(datasets, batch_size=2, shuffle=True)

    # for batch, (x, y) in enumerate(train_dataloader):
    #    print(x.squeeze(1).size())
    #    print(y.size())


# demonstrate()
    