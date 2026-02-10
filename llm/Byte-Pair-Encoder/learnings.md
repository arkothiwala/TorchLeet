1) didn't understand at first why do we need to create a vocab whose key is a tuple
2) instead of just bigrams, I had created ngrams thinking we need all ngrams. upon seeing the test case, I realised that we need to create the n_grams and just find the bigrams

3) during merging step, rather than adding a new key to the vocab, I was replacing old keys with the new keys
4) updated dictionary during the forloop -> throwing error

- I had realised that the exercise provided by the authors isn't done very well
    - Observations:
        - terminology is different [corpus, docs, vocabulary etc.]
        - example input doesn't expose necessary complexity
    - Fix:
        - Changed the example to be in sync with [huggingface BPE tutorial](https://huggingface.co/learn/llm-course/en/chapter6/5)


Open points:
1) How do you decide the number of iterations to run for BPE?
    - Ideally the tokens created by it on the documents should be a significant representation of it.
    - How do we craft the content such a way that the right set of meaningful tokens are learnt [training and tokenization corpus needs to be in sync].
2) How do we tokenize a text given BPE and corpus?
    - Should construct a TRIE data structure for the same?


---
# Learnings from Andrej Karpathy's [GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) video

- Why do we need tokenizer?
    - characters makes it difficult -> we want to compress.
    - using words will lead to out of vocab error too frequently
    - 
- **Lot of LLM performance issues stem from tokenizers**
    - LLM poor at math due to non-deterministic nature of tokenizing amounts [394827 -> 39-4-827]
    - LLMs poor at other languages because poor tokenization learnt for other languages as a consequence of poor training data in that language.
    - vocab part of tokenizer training but not part of LLM training can disrupt LLM behavior
    - GPT2 is bad at python because whitespaces aren't encoded properly [got fixed in GPT4]
    - ending with space " " causes issue ["I am a" -> " boy" v/s "I am a " -> "bo"] -> this causes issue because many tokens contain leading whitespaces -> now we have used that space so LLM is forced to output a token either doesn't start with whitespace [less no of tokens] or it makes final output weird if new token also contain whitespace -> resulting in double whitespace -> model would have hardly seen this in the training data.
- BPE algorithm
- Input to BPE [characters v/s unicodes v/s bytes]
    - string is a sequence of unicodes
    - [sentencepiece](https://github.com/google/sentencepiece) uses unicodes instead of bytes which is used by [tictoken](https://github.com/openai/tiktoken). However, sentencepiece has a fallback to use bytes in case it finds out of vocab token [i.e. UNK]
- How to decide vocab size?
    - It's a `hyperparameter` - GPT2 has 50k vocab and GPT4 has 100k vocab
    - vocab size positively correlates with embeddings space [affects memory] as well as the final softmax output [affects performance]
- **NO MERGE rules**
    - Tokenizer shouldn't combine certain characters with the prior characters - like ephostropys ['m, 'd, 'll, 're etc.], punctuations [., ?, ,, ! etc.], long numbers [more than 3 digit numbers], whitespaces etc.
    - We solve it by splitting the doc based on regex encoding above mentioned NO MERGE rules
- tictokenizer v/s sentencepiece
    - tictokenizer - doesn't provide training script
    - sentencepiece - provides training script but need to watch out for the settings/parameters/flags very carefully.
- Not covered in the video
    - How to curate tokenizer training data
- Can we skip tokenization?
    - There is a path but it doesn't have a lot of momemtum.
    - There is some research done and was explained in the video