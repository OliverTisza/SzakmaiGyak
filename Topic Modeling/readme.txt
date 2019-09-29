------------------------------------------

Amit tartalmaz a Source + Skeleton mappa:

------------------------------------------
-Forráskódokat
-Egy vázat, a scriptek "--path" paraméterével egy mappát adsz meg DE a mappán belül szükségesek további mappák. (pl.: cnet mappa)
-Egy batch fájlt ami megfelelõ sorrendben futtatja le a scripteket, ha valahol hiba van onnantól kezdõdõen beleírja az info.txt-be összecsúsznak a hibaüzenetek, de úgyis csak a legfelsõ a lényeges. (Ha a sorban szerepel a "Traceback" akkor már az a következõ hiba)
A konzol nyitva marad ha végzett a batch.
A konzolban fõleg az íródik ki, hogy amikor végzett egy részfeladattal, az meddig tartott.

-scriptsorrend.txt ha valamiért kellene.

------------------------------------------

Amit NEM tartalmaz a Source + Skeleton mappa DE szükséged lesz rá:

------------------------------------------
-Ritkított szóbeágyazások tömörítve (nekem: glove300d_l_0.1_DL_top400000.emb.gz
					    glove300d_l_0.1_GS_top400000.emb.gz
					    glove300d_l_0.1_kmeans_top400000.emb.gz)

-Cnet kicsomagolva (vagy átírod a cnet.py-t hogy ne kelljen kicsomagolni, vagy megkérsz rá, hogy írjam meg és megvárod míg megírom)
Ezt assertions.csv néven fogja keresni a script (cnet.py 40.sor)

Az én cnet-em: conceptnet-assertions-5.7.0.csv.gz --extract--> assertions.csv

