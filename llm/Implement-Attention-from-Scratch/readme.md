- [Ground truth torch implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention)

# Mistakes made in the early implementation

- didn't implement mask correctly
    - mistake 1 - applied mask after softmax
    - mistake 2 - applied mask as zeros and ones
        - this was bad because other values could have been less than zero and it would make the scores values incorrect
        - had to apply mask with -ve infinity where the values were false
    - mistake 3 [very important]
        - masks are of two types [boolean and float masks [as mentioned in ALiBi - attention with linear biases]]
        - For float mask, instead of adding the masks I was multiplying it [similar to other masks in deep learning]
            - We need multiplicative effect in probability space that why we do addition in logit space as it is already in log space
            - If we do multiplication in log space then it would create exponential effects in probability space [bias>1 likely to blow up the values and bias<1 likely to vanish the values]
            - Also gradiant flows are smoother with addition over multiplication **[gemini conversation details here](https://gemini.google.com/app/b5a09d81a9409ee5?pli=1)**
- initially, forgot applying softmax then forgot doing normalization by sqrt(d_k)