from transformers import BertTokenizer, BertModel
import torch

device = "cuda:0"if torch.cuda.is_available else "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased").to(device)
    

# Use bert model for first layer of encoder
def bert_function(text):
    value = tokenizer(text, return_tensors="pt").to(device)
    output = model(**value)
    hidden_layer = output[0]
    last_hidden_layer = hidden_layer[0, hidden_layer.size(1)-1, :].unsqueeze(-1)
    return last_hidden_layer.squeeze(1).tolist()


def demonstrate():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained("bert-large-uncased")

    text = "Hello world, hello everyone. I hope you can always make your day."

    value = tokenizer(text, return_tensors="pt")
    output = model(**value)
    hidden_layer = output[0]
    print(hidden_layer.size())
    last_hidden_layer = hidden_layer[0, hidden_layer.size(1)-1, :].unsqueeze(-1)

    print(last_hidden_layer.squeeze(1).size())


# demonstrate()