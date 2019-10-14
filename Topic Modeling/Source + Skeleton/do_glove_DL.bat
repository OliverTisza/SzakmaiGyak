@echo off


python sparse_embedding.py --savepath Glove_DL --embedding-location C:\Users\Oliver\Desktop\TModeling\Glove_sparse\glove300d_l_0.1_DL_top400000.emb.gz 2> info.txt
python cnet.py --path Glove_DL --cnet-location C:\Users\Oliver\Desktop\Week_3\feladat_3\conceptnet-assertions-5.7.0.csv.gz 2>> info.txt
python make_docs.py --path Glove_DL 2>> info.txt
python save_dict.py --path Glove_DL 2>> info.txt
python save_corpus.py --path Glove_DL 2>> info.txt
python top_words.py --path Glove_DL 2>> info.txt
python LDA_creator.py --path Glove_DL 2>> info.txt
python matrix_creator.py --path Glove_DL 2>> info.txt
python matrix_plotter.py --path Glove_DL 2>> info.txt

cmd /k



