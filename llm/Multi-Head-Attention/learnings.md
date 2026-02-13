### Learnings:
- validating custom MHA with pytorch isn't super  straight forward.
    - It is due to the weights initialization for Q,K,V and outer/concat layer -> which is random
    - even this wouldn't have been solved with torch.manual_seed() because:
        - **reason 1 - different weights and random initialization** - For linear layers, MHA uses xavier intialization in MHA v/s I was using nn.linear which uses kaiming_uniform initialization
        - **reason 2 - architecture mismatch** - MHA uses single large Wqkv matrix [i.e. in_proj_weight] v/s I was using three seperate nn.Linear layers

- Implementation
    - [bias isn't initialized with zero in Linear layer](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/nn/modules/linear.py#L117) - didn't know linear layer uses kaiming_uniform initialization. I was under the impression that all layers use xavier only. Also bias terms have non-zero initialization in Linear layer.
    - **no bias term in MHA** - didn't know/realise that original attention is all you need paper didn't have bias term in the MHA. However, modern libs like pytorch has bias=True bydefault.
    - **QKV projection inside MHA** - didn't know that Wq, Wk, Wv are part of the MHA [was under the impression that it happens before MHA after PE step]
    - **didn't handle mask** - mask had full shape but we were doing SDPA on a head which has lower dimensions
- torch
    - [view v/s reshape](https://chatgpt.com/share/698f978d-1e88-800a-9657-dde5afe4aea4)
        - error encountered
        `RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.`
        - code snippet?
        ```python
        attention_per_head = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # (B, H, L_k, d_mha)
        attention_per_head = attention_per_head.permute(0,2,1,3) # (B, L_k, H, d_mha)
        attention_concatenated = attention_per_head.view(batch_size, seq_len, self.d_model) # (B, L_k, d_model)
        ```
        - reason:
            - Tensor.view operates on contegious memory
            - Tensor.permute and other reindexing operation makes the view non-contegious
        - question: when to use view v/s reshape
            - torch recommands reshape over view for most scenarios
            - how reshape works - it first try with view -> if it fails then it create copy of the tensor
            - if one wants to use view() -> then do permute().contiguous().view() -> this makes the memory contiguous
            - while reshape doesn't let the code fail on non-contiguous memory like view, it also can lead to silent performance issues related to memory
    - transpose v/s permute
        - transpose swaps only two dimenstions
        - permute can change n_dimensions but it requires exact order [like t.permute(0,2,1,3) v/s t.transpose(1,2)]
        - permute is cleaner and explicit. That's why most libraries use it over transpose [x.transpose(1,2).transpose(2,3).transpose(0,1)]
    - performance - doing single large multiplication is efficient over doing same no multiplications in 3 metrics. That's why MHA has single Wqkv for self attention.

### Roughwork:
<strike>

- break the Q,K,V to the individual head [torch will have an method of splitting a single dimention into multiple ones]
    - input shape (for Q,K,V) = (B, L, D)
    - output shape (for Q,K,V) = (B, H, L, Dh)
- apply scaled dot product attention
    - output shape = (B, H, L_k, D_v)
- concat layer
    - output shape = (B, L_k, D_v*n_heads)
- linear layer
    - W_aa -> (D_model, D_model)
    - output shape = (B, L_k, D_model)
</strike>
