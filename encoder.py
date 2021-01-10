"""Implementation of the encoder."""
import torch


class Encoder(torch.nn.Module):
  """The encoder predicts the scores given the text review."""

  def __init__(self, embeddings, hidden_dim, output_dim):
    super(Encoder, self).__init__()
    embedding_dim = embeddings.shape[1]
    self.embedding_layer = torch.nn.Embedding.from_pretrained(embeddings)
    self.embedding_layer.weight.requires_grad = True

    self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    self.linear = torch.nn.Linear(hidden_dim, output_dim)
    self.activation = torch.nn.Sigmoid()

  def forward(self, inputs):
    x = self.embedding_layer(inputs)
    x = self.activation(x)
    _, hidden = self.rnn(x)
    return self.activation(self.linear(hidden))
