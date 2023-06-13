from torch import nn

class ENCODER(nn.Module):
    """Some Information about ENCODER"""
    def __init__(self, embedding_feature):
        super(ENCODER, self).__init__()

        self.sequence, self.feature = 140, 1
        self.embedding = embedding_feature
        self.hidden = embedding_feature * 2

        # Sequence 정보를 LSTM을 통과시킴
        self.encoder_1 = nn.LSTM(input_size = self.feature, hidden_size = self.hidden, num_layers = 1, batch_first = True)
        # 통과된 정보를 특정 차원으로 줄이면서 Bottle Neck구조를 가짐
        self.encoder_embedding = nn.LSTM(input_size = self.hidden, hidden_size = self.embedding, num_layers = 1, batch_first = True)

    def forward(self, x):
        # 출력 데이터 형식이 (Batch, Sequence) 라서 Feature 부분을 뒤에 붙여줌
        x = x.unsqueeze(dim = -1)
        # 데이터 LSTM 통과
        x, (_, _) = self.encoder_1(x)
        # 저차원 embedding
        x, (hidden_state, cell_state) = self.encoder_embedding(x)
        # hidden_state 반환
        return hidden_state
    
class DECODER(nn.Module):
    """Some Information about DECODER"""
    def __init__(self, embedding_feature,):
        super(DECODER, self).__init__()

        self.sequence, self.feature = 140, 1
        self.embedding = embedding_feature
        self.hidden = embedding_feature * 2

        self.decoder_1 = nn.LSTM(input_size = self.embedding, hidden_size = self.embedding, num_layers = 1, batch_first = True)
        self.decoder_2 = nn.LSTM(input_size = self.embedding, hidden_size = self.hidden, num_layers = 1, batch_first = True)
        self.decoder_3 = nn.Linear(in_features = self.hidden, out_features = self.feature)

    def forward(self, x):
        # Encoder의 hidden_state(1, batch, embedding)를 받아서 Decoder_1에 넣어서 해독
        # -> Repeat을 통해서 sequence개 만큼 복사 -> Sequence 층 생성
        x = x.repeat((1, 1, self.sequence))
        x = x.reshape((x.shape[1], self.sequence, -1))
        x, (_, _) = self.decoder_1(x)
        # 출력값을 한번 더 해독
        x, (_, _) = self.decoder_2(x)
        # 마지막 출력값을 linear를 통해서 reconstruct
        x = self.decoder_3(x)
        # 입력과 차원의 수 맞추기 위해서 squeeze
        return x.squeeze()
    
class MODEL(nn.Module):
    """Some Information about MODEL"""
    def __init__(self, embedding_feature):
        super(MODEL, self).__init__()

        self.encoder = ENCODER(embedding_feature)
        self.decoder = DECODER(embedding_feature)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x