@echo off



python semcat_prepare.py 2> info.txt
python cnet.py --path SemCat --cnet-location C:\Users\Oliver\Desktop\Week_3\feladat_3\conceptnet-assertions-5.7.0.csv.gz 2>> info.txt
python make_docs.py --path SemCat 2>> info.txt
python save_dict.py --path SemCat 2>> info.txt
python save_corpus.py --path SemCat 2>> info.txt
python top_words.py --path SemCat 2>> info.txt
python LDA_creator.py --path SemCat 2>> info.txt
rem python matrix_creator.py --path SemCat 2>> info.txt
rem python matrix_plotter.py --path SemCat 2>> info.txt

cmd /k



