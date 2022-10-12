# LAION-dedup

The goal of this repository is to provide different tools for deduplication, which can be used to:

- Know how many unique images in the dataset
- Reduce memorization in text to image models
- Increase training efficiency
- Study the impact of the pre-training data's near-duplicates on the performance of downstream tasks

## Resources

### Software 

- Locality-sensitive hashing (LSH) with FAISS: <https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexLSH.html>
- PHash <https://www.phash.org>, near-duplicate detection based on discrete cosine transform


### Datasets for evaluation

- Copydays: <https://lear.inrialpes.fr/~jegou/data.php>
