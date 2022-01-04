import subprocess


spacy_nlp = [
    "fr_core_news_sm",
    "fr_core_news_md",
    "fr_core_news_lg"
]
spacy_embedding = [
    "fr_core_news_md",
    "fr_core_news_lg"
]

for snlp in spacy_nlp:
    for sembedding in spacy_embedding:
        subprocess.run(["python",
                        "./src/supervised/train.py",
                        f"./data/processed/train_{snlp}_{sembedding}.npy",
                        "random_forest",
                        f"model_out/random_forest_{snlp}_{sembedding}.txt"])
