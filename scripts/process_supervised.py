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
                        "./src/process/process_supervised.py",
                        "spacy",
                        "--spacy_nlp",
                        snlp,
                        "--spacy_embedding",
                        sembedding])
