import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNGLU(nn.Module):
    """
    CNN with Gated Linear Units and attention pooling.
    Works well on smaller datasets.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)

        self.glu = nn.GLU(dim=1)

        self.attention = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(48, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        c3 = self.glu(self.conv3(x))
        c5 = self.glu(self.conv5(x))
        c7 = self.glu(self.conv7(x))

        combined = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)  # (batch, seq_len, 96)
        attn_weights = F.softmax(self.attention(combined), dim=1)  # (batch, seq_len, 1)
        weighted = torch.sum(attn_weights * combined, dim=1)  # (batch, 96)

        logits = self.classifier(weighted)
        return logits, attn_weights

