import torch
import torch.nn.functional as F


class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V):
        # Calculate the dot products between Q and K
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply the attention weights with V to get the output
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.out_linear = torch.nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V):
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        out, _ = self.attention.forward(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model)
        out = self.out_linear(out)

        return out

# Example usage
d_k = 64
attention = ScaledDotProductAttention(d_k)

# Random tensors representing Q, K, and V
Q = torch.rand(10, 20, d_k)  # (batch_size, seq_length, d_k)
K = torch.rand(10, 20, d_k)
V = torch.rand(10, 20, d_k)

output, attention_weights = attention.forward(Q, K, V)

#print("Output:", output)
print("Attention Weights:", attention_weights)

# Example usage
d_model = 128
num_heads = 8
multi_head_attention = MultiHeadAttention(d_model, num_heads)

# Random tensors representing the input sequence (patch embeddings)
patch_embeddings = torch.rand(10, 16, d_model)  # (batch_size, num_patches, d_model)

output = multi_head_attention.forward(patch_embeddings, patch_embeddings, patch_embeddings)

print("Multi-Head Attention Output:", output)
