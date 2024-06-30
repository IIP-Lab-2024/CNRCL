import os
os.environ['CURL_CA_BUNDLE'] = ''

from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')

def read_news_file(file_path):
    news_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.strip().split('\t')
            if len(row) == 8:
                news_data.append(row)
    return pd.DataFrame(news_data, columns=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]).dropna()

train_news_file = "./MIND-small/train/news.tsv"
test_news_file = "./MIND-small/test/news.tsv"
dev_news_file = "./MIND-small/dev/news.tsv"

train_news_df = read_news_file(train_news_file)
test_news_df = read_news_file(test_news_file)
dev_news_df = read_news_file(dev_news_file)

news_df = pd.concat([train_news_df, test_news_df, dev_news_df]).drop_duplicates(subset=['news_id']).reset_index(drop=True)
news_df['combined_text'] = news_df['title'] + " [SEP] " + news_df['abstract']

def get_bert_embeddings(news_df, column_name):
    bert_embeddings = {}
    n = 1
    for _, news in news_df.iterrows():
        print(f"processing news for news {n} {column_name}...")
        text = news[column_name]
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        bert_embedding = outputs.last_hidden_state[0, 0].detach().numpy()
        bert_embeddings[n] = bert_embedding
        n = n + 1
    return bert_embeddings

combined_embeddings = get_bert_embeddings(news_df, 'combined_text')
combined_embeddings = {0: [0.0] * 768, **combined_embeddings}
df_combined = pd.DataFrame.from_dict(combined_embeddings, orient='index')
df_combined = df_combined.reset_index()
df_combined.columns = ['news_id'] + [f'vector_{i}' for i in range(df_combined.shape[1] - 1)]
df_combined.to_csv("./news_vectors_small.csv", index=False)

# 初始化相似度结果字典
similarity_results = {}


def compute_similarity_from_vectors(vector1, vector2):
    if vector1 is not None and vector2 is not None:
        similarity = cosine_similarity([vector1], [vector2])[0][0]
    else:
        similarity = 0.0
    return similarity



# for each impression
with open('./MIND-small/train/behaviors.tsv', 'r', encoding='utf-8') as train_behaviors_f:
    # load
    news_vectors = pd.read_csv('./news_vectors_small.csv')
    news_vector_dict = {row['news_id']: row[1:].values for _, row in news_vectors.iloc[1:].iterrows()}

    for behavior_index, line in enumerate(train_behaviors_f):

        if behavior_index % 100 == 0:
            print(f"Processed {behavior_index} impressions")

        impression_ID, user_ID, time, history, impressions = line.split('\t')
        history_news = history.strip().split(' ')
        non_click_impressions = [impression[:-2] for impression in impressions.strip().split(' ') if
                                 impression[-2:] != '-1']

        # compute user vectors
        history_vectors = [news_vector_dict[news_id] for news_id in history_news if news_id in news_vector_dict]
        if not history_vectors:
            default_similarity = 0
            for non_click_impression in non_click_impressions:
                if impression_ID not in similarity_results:
                    similarity_results[impression_ID] = {}
                similarity_results[impression_ID][non_click_impression] = default_similarity
            continue
        history_vector = np.mean(history_vectors, axis=0)
        for non_click_impression in non_click_impressions:
            if non_click_impression in news_vector_dict:
                if impression_ID not in similarity_results:
                    similarity_results[impression_ID] = {}
                similarity = compute_similarity_from_vectors(history_vector, news_vector_dict[non_click_impression])
                similarity_results[impression_ID][non_click_impression] = similarity
# load
with open('./similarity_results_history_small.pkl', 'wb') as f:
    pickle.dump(similarity_results, f)

