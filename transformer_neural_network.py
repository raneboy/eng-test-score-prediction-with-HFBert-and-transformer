import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import Linear, TransformerDecoder, TransformerDecoderLayer, Softmax
from preprocess_data import preprocess_data, CustomStrokeDataset
from torch.utils.data import Dataset, DataLoader


device = "cuda:0"if torch.cuda.is_available else "cpu"

class TransformerNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Positional Encoder
        self.position_encoder = PositionalEncoding()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=8)

        # Encoder
        decoder_layer = TransformerDecoderLayer(d_model=1024, nhead=4, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=8)
        

        # Linear laer
        self.linear_layer = Linear(1024, 11)
        self.softmax = Softmax(dim=1)

        # Randomly initilize the linear layer bias
        self.init_bias_weight()


    def init_bias_weight(self):
        initrange = 0.2
        self.linear_layer.bias.data.zero_()
        self.linear_layer.weight.data.uniform_(-initrange, initrange)


    def forward(self, x: Tensor) -> Tensor:
        x = self.position_encoder(x)
        encoder_output = self.transformer_encoder(x)

        decoder_input = torch.zeros(encoder_output.size(0), 6, 1024).to(device)
        decoder_output = self.transformer_decoder(encoder_output, decoder_input)

        linear_output = self.linear_layer(decoder_output)
        output = self.softmax(linear_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        pe = torch.zeros(32, 6, 1024)
        position = torch.arange(6).unsqueeze(1)

        div = ( 10000 ** ( (2 * torch.arange(0, 1024, 2)) / 1024) )

        pe[:, :, 0::2] = torch.sin(position / div)
        pe[:, :, 1::2] = torch.cos(position / div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(0) != 32:
            pe = self.pe[:x.size(0), :, :]
        else:
            pe = self.pe
        output = x + pe.to(device)
        return output


# This function used to test and debug purpose
def demonstrate_network():

    processed_data = preprocess_data("Feedback Prize - English Language Learning\\data\\train.csv", True)

    train_dataset = CustomStrokeDataset(processed_data)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    for x, y in train_dataloader:
        x = x
        y = y

    x = x[3].to("cpu")
    y = y[3].to("cpu")

    # Transformer encoder layer
    encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=2, batch_first=True)
    transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)

    # Transformer decoder layer
    decoder_layer = TransformerDecoderLayer(d_model=1024, nhead=2, batch_first=True)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=2)

    # Linear layer and sigmoid for classification
    linear_layer = Linear(1024, 5)
    softmax = Softmax(dim=1)

    # Lost function
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------- #
    # Input and output of transformer encoder
    src = x
    out = transformer_encoder(src)
    print(out)
 
    # Input and output of tranformer decoder
    src2 = torch.zeros(32, 6, 1024)
    out2 = transformer_decoder(src2, out)

    # output of Linear layer
    src3 = out2 
    out3 = linear_layer(out2)
    out3 = softmax(out3)

    # The loss function
    label = [[[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]], [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
             [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]], [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]]]
    out4 = torch.tensor(out3)
    label = torch.tensor(y, dtype=torch.float)
    loss = criterion(out4, label)

    # Print all output
    print()
    print("Output :")
    print("=" * 50)
    print("Encoder input size : ", src.size())
    print("Encoder output size : ", out.size())
    print("=" * 50)
    print("Decoder input size : ", src2.size())
    print("Decoder output size : ", out2.size())
    print("=" * 50)
    print("Linear input size : ", src3.size())
    print("Linear output size : ", out3.size())
    print("=" * 50)
    print("Loss :")
    print(loss.item() * 1000)
    print("=" * 50)
    print()


# This function used to test the tranformer model
def test_model():
    x = torch.rand(7, 6, 1024)
    # pe = PositionalEncoding()
    # y = pe(x)
    x = x.to(device)
    model = TransformerNetwork().to(device=device)

    output = model(x)
    print(output.size())


# demonstrate_network()
# test_model()
