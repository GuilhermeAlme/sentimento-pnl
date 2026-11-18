import pandas as pd
import os
import re
os.makedirs('resultados', exist_ok=True)
df = pd.read_csv('data/avaliacoes.csv')
print(len(df))
print(df['sentimento'].value_counts())
n_palavras = df['texto'].str.split().str.len()
print(n_palavras)
df['n_palavras'] = n_palavras
print(df.groupby('sentimento')['n_palavras'].mean())
print(df[df['sentimento']=='positivo']['texto'].head(3))
print(df[df['sentimento']=='negativo']['texto'].head(3))

with open('resultados/exploracao.txt', 'w', encoding='utf-8') as arq:
    print('Total de linhas:', len(df), file=arq)
    print(df['sentimento'].value_counts(), file=arq)
    print(df.groupby('sentimento')['n_palavras'].mean(), file=arq)
    print(df[df['sentimento']=='positivo']['texto'].head(3), file=arq)
    print(df[df['sentimento']=='negativo']['texto'].head(3), file=arq)





