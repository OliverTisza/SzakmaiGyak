------------------------------------------

Amit tartalmaz a Source + Skeleton mappa:

------------------------------------------
-Forr�sk�dokat
-Egy v�zat, a scriptek "--path" param�ter�vel egy mapp�t adsz meg DE a mapp�n bel�l sz�ks�gesek tov�bbi mapp�k. (pl.: cnet mappa)
-Egy batch f�jlt ami megfelel� sorrendben futtatja le a scripteket, ha valahol hiba van onnant�l kezd�d�en bele�rja az info.txt-be �sszecs�sznak a hiba�zenetek, de �gyis csak a legfels� a l�nyeges. (Ha a sorban szerepel a "Traceback" akkor m�r az a k�vetkez� hiba)
A konzol nyitva marad ha v�gzett a batch.
A konzolban f�leg az �r�dik ki, hogy amikor v�gzett egy r�szfeladattal, az meddig tartott.

-scriptsorrend.txt ha valami�rt kellene.

------------------------------------------

Amit NEM tartalmaz a Source + Skeleton mappa DE sz�ks�ged lesz r�:

------------------------------------------
-Ritk�tott sz�be�gyaz�sok t�m�r�tve (nekem: glove300d_l_0.1_DL_top400000.emb.gz
					    glove300d_l_0.1_GS_top400000.emb.gz
					    glove300d_l_0.1_kmeans_top400000.emb.gz)

-Cnet kicsomagolva (vagy �t�rod a cnet.py-t hogy ne kelljen kicsomagolni, vagy megk�rsz r�, hogy �rjam meg �s megv�rod m�g meg�rom)
Ezt assertions.csv n�ven fogja keresni a script (cnet.py 40.sor)

Az �n cnet-em: conceptnet-assertions-5.7.0.csv.gz --extract--> assertions.csv

