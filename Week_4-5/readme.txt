Ezeket a dolgokat biztosan át kell írnod hogy mûködjön

Útvonalak:

	semcor.data.xml-hez: 29.sor (mindketto fajlban)
	mallet-hez: 72.-73.sor


Megyjegyzés(ek):

- A pyLDAvis használatához bele kellett írnom a 
C:\Users\[USERNAME]\AppData\Local\Programs\Python\Python35-32\Lib\site-packages\pyLDAvis\_prepare.py
257.sorában hogy 'sort=True' és csak úgy mûködött 

pd.concat([default_term_info] + list(topic_dfs),sort=True)

pandas nem tud hianyos dolgokat concatolni alapbol vagy valami ilyesmi baja volt
	
 
- A mallet-et is szerettem volna visualizalni viszont 'inference' attributuma nem volt ezert atkonvertaltam lda-va,
ezek utan beleutkozik 2-3 0-val valo osztasba, de nem dob errort es utana meg is tekintheto a pyLDAvis

- Az egesz kodot egy main()-be kellett wrappelnem mert windows-on tobb szalnal rekurzioba esik, ez coherencia szamitasnal jott elo


