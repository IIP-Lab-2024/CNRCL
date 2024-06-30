import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import multiprocessing as mp


news_df = pd.read_csv("./MIND-small/train/news.tsv", delimiter="\t", header=None, names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]).dropna()

glove_file = "./glove.840B.300d.txt"
with open(glove_file, 'rb') as f:
    word_vectors = {}
    for line in f:
        line = line.strip()
        word, vector = line.split(maxsplit=1)
        vector = np.fromstring(vector, sep=' ')
        word_vectors[word] = vector
print(word_vectors)

def news_similarity(news1, news2):
    news1_words = [w.lower() for w in news1["title"].split() + news1["abstract"].split() if w.lower() in word_vectors]
    news2_words = [w.lower() for w in news2["title"].split() + news2["abstract"].split() if w.lower() in word_vectors]

    for w in news1_words + news2_words:
        if w not in word_vectors:
            print(w)
    if news1_words and news2_words:
        news1_embedding = np.mean([word_vectors[w] for w in news1_words], axis=0)
        news2_embedding = np.mean([word_vectors[w] for w in news2_words], axis=0)
        similarity = cosine_similarity([news1_embedding], [news2_embedding])[0][0]
    else:
        similarity = 0.0
    return similarity


n_similar = 10  #

# Define a function to calculate similarities between a news item and all other news items
def process_news(news_idx, news_df, word_vectors):
    news_similarities = []
    for j, news2 in news_df.iterrows():
        if news_idx != j:
            similarity = news_similarity(news_df.iloc[news_idx], news2, word_vectors)
            news_similarities.append((news_df.iloc[j]["news_id"], similarity))
    news_similarities.sort(key=lambda x: x[1], reverse=True)
    return (news_df.iloc[news_idx]["news_id"], [x[0] for x in news_similarities[:n_similar]])

# Define a function to parallelize the computation of similarities
def parallel_process_news(news_df, word_vectors, n_jobs=8):
    with mp.Pool(n_jobs) as pool:
        news_similarities = pool.starmap(process_news, [(i, news_df, word_vectors) for i in range(news_df.shape[0])])
    return pd.DataFrame(news_similarities, columns=["news_id", "most_similar_news"])

# Run the parallelized computation
result_df = parallel_process_news(news_df, word_vectors, n_jobs=8)
result_df.to_csv("./similarities_small.csv", index=False)