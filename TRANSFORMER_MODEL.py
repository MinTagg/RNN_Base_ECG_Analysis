from torch import nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        
        # 기존의 입력 데이터는 1개의 feature를 가지지만, Transformer의 Head의 수를 증가시키기 위해서 값을 복사한다
        x = x.unsqueeze(-1).repeat((1, 1, self.input_dim))
        # 인코딩
        x_encoded = self.encoder(x.transpose(0, 1))
        # 디코딩
        x_decoded = self.decoder(x_encoded, x_encoded)
        
        # 디코더의 결과값을 Linear layer를 통해서 입력의 1개 feature와 동일한 결과를 만들도록 한다
        x_reconstructed = self.fc(x_decoded.transpose(0, 1))
        
        return x_reconstructed.squeeze(-1)