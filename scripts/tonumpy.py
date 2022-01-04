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
                        "./src/process/tonumpy.py",
                        "--inp",
                        f"./data/processed/train_{snlp}_{sembedding}.csv"])
        subprocess.run(["python",
                        "./src/process/tonumpy.py",
                        "--inp",
                        f"./data/processed/test_{snlp}_{sembedding}.csv"])
