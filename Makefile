build:
	mkdir -p report/images
	mkdir -p data/clean
	mkdir -p data/processed
	mkdir -p models
	unzip data/raw/datasets.zip -d data/raw
	wget https://zenodo.org/record/3241447/files/frwiki-20181020.treetag.2.ngram-pass2__2019-04-08_09.02__.s500_w5_skip.word2vec.bin?download=1 -P data/word_embedding
	wget https://zenodo.org/record/3241447/files/frwiki-20181020.treetag.2__2019-01-24_10.41__.s500_w5_skip.word2vec.bin?download=1 -P data/word_embedding
	mv data/word_embedding/frwiki-20181020.treetag.2.ngram-pass2__2019-04-08_09.02__.s500_w5_skip.word2vec.bin\?download=1 data/word_embedding/word2vec_pre_process.bin
	mv data/word_embedding/frwiki-20181020.treetag.2__2019-01-24_10.41__.s500_w5_skip.word2vec.bin\?download=1 data/word_embedding/word2vec_no_pre_process.bin
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz -P data/word_embedding
	gunzip data/word_embedding/cc.fr.300.bin.gz
	python -m spacy download fr_core_news_sm
	python -m spacy download fr_core_news_md
	python -m spacy download fr_core_news_lg
	python -m spacy download fr_dep_news_trf

clean:
	python src/process/clean.py

pre_explore:
	python src/viz/pre_explore.py

process:
	python src/process/process.py
