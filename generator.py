"""Implementation of the encoder."""
import torch


class Generator(torch.nn.Module):
  """The generator selects the parts of the text review used for the encoder."""

  def __init__(self, embeddings, hidden_dim):
    super(Generator, self).__init__()
    embedding_dim = embeddings.shape[1]
    self.embedding_layer = torch.nn.Embedding.from_pretrained(embeddings)
    self.embedding_layer.weight.requires_grad = False

    self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.linear = torch.nn.Linear(2 * hidden_dim, 1)
    self.activation = torch.nn.Sigmoid()

  def forward(self, inputs):
    x = self.embedding_layer(inputs)
    x = self.activation(x)
    out, hidden = self.rnn(x)
    return self.activation(self.linear(out))

  def select(self, logits):
    out = (logits > 0.5).long()
    return out

  def loss(self, selection, inputs):
    selection = selection.float()
    selection_cost = torch.mean(torch.sum(selection, dim=1))
    l_padded_mask = torch.cat([selection[:, 0].unsqueeze(1), selection], dim=1)
    r_padded_mask = torch.cat([selection, selection[:, -1].unsqueeze(1)], dim=1)
    continuity_cost = torch.mean(
        torch.sum(torch.abs(l_padded_mask - r_padded_mask), dim=1))
    return selection_cost, continuity_cost
