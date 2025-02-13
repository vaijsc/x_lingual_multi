from datasets import load_dataset, Dataset
import os
import json
from sentence_transformers import SentenceTransformer, util
from loguru import logger
import numpy as np
import random
import itertools

NUM_SAMPLES=50_000

def input_template(article: str, input_prefix: str = "###\nArticle: ", input_suffix: str = "\n\n", output_command: str = "Summarize the above article in 3 sentences.") -> str:
    return f"{input_prefix}{article}{input_suffix}{output_command}"

if __name__ == "__main__":
    sent_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')

    corpus_sentences_dict = {}
    corpus_sentences_dict.update(dict(zip(dataset['article'], dataset)))
    corpus_sentences = list(corpus_sentences_dict.keys())
    logger.info(f"Create corpus with {len(corpus_sentences)} documents")

    corpus_embeddings = sent_model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    logger.success(f'Encode successfully')

    clusters = util.community_detection(
        corpus_embeddings, 
        min_community_size=10, 
        threshold=0.75
    )
    total_num = 0
    for cluster in clusters:
        total_num += len(cluster)
    logger.info(f'There are in total {total_num} elements in {len(clusters)} clusters')

    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        cluster_dict[i] = [corpus_sentences_dict[corpus_sentences[sentence_id]] for sentence_id in cluster]
    
    clusers_len = np.array([len(c) for c in clusters]) # array of length (= number of element) of each cluster
    cluster_num_sample = clusers_len * NUM_SAMPLES / sum(clusers_len)
    cluster_num_sample = np.where(cluster_num_sample<1,2,cluster_num_sample.astype(int)+1) # array stores number of element in each cluster should be sampled

    res = []
    ids = []
    for i, cluster in enumerate(clusters):
        examples = cluster_dict[i]

        sample_index = random.sample(list(range(len(cluster))), min(cluster_num_sample[i], len(cluster)))
        
        ids.append([cluster[i] for i in sample_index])
        res.append([examples[i] for i in sample_index])

    res = list(itertools.chain.from_iterable(res))
    logger.info(f'Sample {len(res)} documents.')

    cluster_hf_data = Dataset.from_list(res).shuffle(seed=42).select(range(50_000))
    cluster_hf_data = cluster_hf_data.train_test_split(test_size=0.02)
    cluster_hf_data['valid'] = cluster_hf_data.pop('test')

    os.makedirs("data/cnn/cluster", exist_ok=True)

    num = 0
    for split in ['train', 'valid']:
        with open(f"data/cnn/cluster/{split}.jsonl", "w+") as fout:
            for data in cluster_hf_data[split]:
                inp = input_template(data['article'])
                out = data['highlights']
                json_data = {'prompt': inp, 'output': out}
                fout.write(json.dumps(json_data) + "\n")
                num += 1

    logger.info(f"Number of lines: {num}")