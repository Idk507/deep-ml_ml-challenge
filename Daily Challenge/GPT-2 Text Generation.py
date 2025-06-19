"""
GPT-2 Text Generation

Implement a Simplified GPT-2-like Text Generation Function
You are tasked with implementing a simplified GPT-2-like text generation function in Python. This function will incorporate the following components of a minimal GPT-2 architecture:

Token Embeddings: Map input tokens to dense vector representations.
Positional Embeddings: Add positional information to token embeddings.
Multi-head Attention: Attend to various parts of the sequence.
Feed-Forward Network: Process attention outputs through a dense layer.
Layer Normalization: Stabilize the training process.
The function must take in the following parameters:

Prompt: The initial text to guide the generation process.
Number of Tokens to Generate: Specify how many tokens to output.
Your function should output the generated text.

Additionally, utilize the helper function load_encoder_hparams_and_params to retrieve:

A dummy encoder.
Model hyperparameters.
Model parameters.
Build your text generation logic around these components. This exercise is designed to help you understand the core concepts behind GPT-2's autoregressive text generation.

Example:
Input:
prompt="hello", n_tokens_to_generate=5
Output:
world <UNK> <UNK> <UNK> <UNK>
Reasoning:
The function encodes the input "hello" into tokens using the dummy encoder, then runs a simplified GPT-2 forward pass to generate 5 tokens. Finally, it decodes the generated tokens back into text.

"""
"""
Understanding Transformer Architecture and Text Generation
Transformers have revolutionized the field of Natural Language Processing (NLP) with their efficient and scalable architecture. This guide provides an in-depth look into the core components of transformers and how they facilitate advanced text generation.

1. Introduction to Transformers
Transformers are a groundbreaking neural network architecture that has significantly advanced NLP. Introduced in the seminal paper "Attention is All You Need" by Vaswani et al. (2017), transformers have outperformed traditional models like Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) in various NLP tasks.

Key Advantages of Transformers
Parallel Processing:
Unlike RNNs, which process input sequences sequentially, transformers handle entire sequences simultaneously. This parallelism leads to substantial improvements in training speed and efficiency.

Scalability:
Transformers can effectively scale to handle large datasets and complex tasks, making them ideal for applications like language translation, text generation, and summarization.

Self-Attention Mechanism:
The core innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence relative to each other. This capability enables the model to capture long-range dependencies and contextual relationships within the text.

Applications of Transformers
Text Generation: Creating coherent and contextually relevant text based on a given prompt.
Machine Translation: Translating text from one language to another with high accuracy.
Text Summarization: Condensing long documents into concise summaries while retaining key information.
Question Answering: Providing accurate answers to user queries based on contextual understanding.
2. Core Concepts
To fully grasp the transformer architecture, it's essential to understand its foundational components. Below are the core concepts that constitute the building blocks of transformers:

2.1 GELU Activation Function
The Gaussian Error Linear Unit (GELU) is an advanced activation function that enhances the performance of deep neural networks.

Mathematical Expression:

GELU
(
x
)
=
0.5
⋅
x
⋅
(
1
+
tanh
⁡
(
2
π
⋅
(
x
+
0.044715
⋅
x
3
)
)
)
GELU(x)=0.5⋅x⋅(1+tanh( 
π
2
​
 
​
 ⋅(x+0.044715⋅x 
3
 )))
Purpose:
GELU introduces non-linearity in the network while maintaining smooth gradient flow. Unlike the Rectified Linear Unit (ReLU) or Sigmoid functions, GELU provides a probabilistic approach to activation, allowing for better handling of uncertainty and improving model performance in deep architectures.

Benefits:

Smooth Activation: Reduces the likelihood of "dead neurons" that can occur with ReLU.
Improved Gradient Flow: Facilitates more stable and efficient training by preventing gradient vanishing or exploding.
2.2 Softmax for Attention
Softmax is a fundamental function used to convert raw attention scores into a probability distribution, ensuring that the weights sum to one.

Mathematical Expression:

Softmax
(
x
i
)
=
exp
⁡
(
x
i
)
∑
j
=
1
n
exp
⁡
(
x
j
)
Softmax(x 
i
​
 )= 
∑ 
j=1
n
​
 exp(x 
j
​
 )
exp(x 
i
​
 )
​
 
Purpose:
In the context of attention mechanisms, Softmax normalizes the attention scores, enabling the model to focus on relevant parts of the input sequence by assigning higher weights to more important tokens.

Example:
If the attention scores for a sentence are [2, 1, 0.1], applying Softmax will convert these to probabilities like [0.659, 0.242, 0.099], indicating the relative importance of each token.

2.3 Layer Normalization
Layer normalization stabilizes and accelerates the training process by standardizing the inputs across the features.

Mathematical Expression:

LayerNorm
(
x
)
=
g
⋅
x
−
μ
σ
2
+
ϵ
+
b
LayerNorm(x)=g⋅ 
σ 
2
 +ϵ
​
 
x−μ
​
 +b
Where:

( \mu ): Mean of input ( x ) along the last axis.
( \sigma^2 ): Variance of ( x ).
( g, b ): Learnable scaling and bias parameters.
( \epsilon ): A small constant to prevent division by zero.
Purpose:
By normalizing the inputs, layer normalization ensures that each layer receives inputs with a consistent distribution, which enhances training stability and convergence speed.

2.4 Multi-Head Attention
Multi-head attention is an extension of the attention mechanism that allows the model to focus on different representation subspaces simultaneously.

Components:

Query (Q), Key (K), Value (V) Matrices: Each attention head computes its own set of Q, K, and V matrices by projecting the input embeddings into different subspaces.
Scaled Dot-Product Attention:
Attention
(
Q
,
K
,
V
)
=
Softmax
(
Q
K
⊤
d
k
)
V
Attention(Q,K,V)=Softmax( 
d 
k
​
 
​
 
QK 
⊤
 
​
 )V
Benefits:

Diversity of Attention: Allows the model to focus on different parts of the input simultaneously.
Enhanced Representation: Captures richer features by aggregating multiple attention heads.
2.5 Feedforward Network (FFN)
The Feedforward Network is a simple yet powerful component applied to each position independently within the transformer.

Mathematical Expression:

FFN
(
x
)
=
Linear
2
(
GELU
(
Linear
1
(
x
)
)
)
FFN(x)=Linear 
2
​
 (GELU(Linear 
1
​
 (x)))
Structure:

First Linear Layer: Projects the input to a higher-dimensional space.
GELU Activation: Introduces non-linearity to the model.
Second Linear Layer: Projects the data back to the original dimensionality.
Purpose:
The FFN enhances the model's capacity to learn intricate patterns.

2.6 Transformer Block
A transformer block is the fundamental building unit of the transformer architecture, combining multi-head attention and the feedforward network with residual connections and layer normalization.

Structure:

Multi-Head Attention Layer:
x
1
=
LayerNorm
(
x
+
MHA
(
x
)
)
x 
1
​
 =LayerNorm(x+MHA(x))
Feedforward Network:
x
2
=
LayerNorm
(
x
1
+
FFN
(
x
1
)
)
x 
2
​
 =LayerNorm(x 
1
​
 +FFN(x 
1
​
 ))
Advantages:

Deep Architecture Support: Facilitates the construction of deep networks without significant performance degradation.
Modularity: Each transformer block can be stacked multiple times, allowing for scalable model depth.
2.7 GPT-2 Text Generation
GPT-2 (Generative Pre-trained Transformer 2) leverages the transformer architecture for generating human-like text. Developed by OpenAI, GPT-2 has demonstrated remarkable capabilities in various NLP tasks.

Key Components:

Word and Positional Embeddings: Captures semantic meaning and token position in a sequence.
Causal Attention: Ensures left-to-right text generation by masking future tokens.
Stacked Transformer Blocks: Refines input representations iteratively.
Text Generation Process:

Provide a prompt to initiate the process.
Tokenize the input into embeddings.
Process embeddings through transformer blocks.
Generate the next token based on probabilities.
Repeat steps 3-4 to produce coherent text.
Conclusion
Transformers have fundamentally transformed NLP by introducing efficient and scalable architectures capable of handling complex language tasks. Understanding their core components such as GELU activation, Softmax attention, layer normalization, multi-head attention, feedforward networks, and the transformer block provides a foundation for leveraging these models in various applications. GPT-2 exemplifies the transformative power of these architectures while highlighting ethical considerations for their use.

"""
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    np.random.seed(42)  # Set the random seed for reproducibility
    encoder, hparams, params = load_encoder_hparams_and_params()
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text
def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 12
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params
