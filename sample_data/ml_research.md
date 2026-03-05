# Machine Learning Research Notes

## Transformer Architecture

The transformer architecture, introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017), fundamentally changed the field of natural language processing. Unlike recurrent neural networks, transformers process all tokens in a sequence simultaneously using a mechanism called self-attention. This allows the model to capture long-range dependencies without the vanishing gradient problem that plagued RNNs.

The core of a transformer is the multi-head attention mechanism. For each attention head, the model computes queries, keys, and values from the input embeddings. The attention score between two tokens is computed as the dot product of their query and key vectors, scaled by the square root of the embedding dimension, and passed through a softmax function.

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines a language model with an external knowledge retrieval system. Instead of relying solely on parameters baked into the model during training, RAG retrieves relevant documents from an external corpus at inference time and conditions the generation on those documents.

The typical RAG pipeline has two phases. In the indexing phase, documents are chunked, embedded using a dense retrieval model, and stored in a vector database. In the inference phase, the user query is embedded and used to retrieve the top-k most similar document chunks. The retrieved chunks are then prepended to the prompt as context for the language model.

RAG significantly reduces hallucinations for knowledge-intensive tasks because the model can ground its answers in retrieved evidence rather than relying on potentially outdated or incorrect memorized facts.

## Vector Databases

A vector database is a data store optimized for storing and querying high-dimensional embedding vectors. The primary operation is approximate nearest neighbor (ANN) search: given a query vector, find the k vectors in the database that are most similar according to a distance metric such as cosine similarity or Euclidean distance.

Popular vector databases include Pinecone, Weaviate, Qdrant, and Chroma. Many use the HNSW (Hierarchical Navigable Small World) algorithm, which builds a multi-layer graph structure that enables logarithmic-time approximate nearest neighbor queries with very high recall.

## Fine-tuning vs Prompting

Fine-tuning involves updating the weights of a pre-trained model on a task-specific dataset. It generally yields better performance on specialized tasks but requires labeled training data, compute resources, and careful management of overfitting. Changes to the underlying task may require re-training.

Prompt engineering, by contrast, involves crafting natural language instructions that guide a frozen model toward the desired behavior. Techniques like few-shot prompting, chain-of-thought prompting, and instruction tuning have dramatically narrowed the performance gap between fine-tuned and prompted models for many tasks.

## Evaluation Metrics

Evaluating language models is notoriously difficult. Common metrics include perplexity (a measure of how well the model predicts a held-out test set), BLEU and ROUGE scores (for generation tasks with reference outputs), and human evaluation. For retrieval tasks, recall@k and mean reciprocal rank (MRR) are standard. For RAG systems, faithfulness and answer relevancy are critical dimensions evaluated by frameworks like RAGAS.
