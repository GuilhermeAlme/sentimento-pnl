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

def preprocessar(texto, remover_stopwords):
     texto = texto.lower()
     palavras = re.findall(r"[a-záàâãéêíóôõúçü]+", texto)
     stopwords = ['a', 'o', 'e', 'de', 'da', 'do', 'que', 'em', 'para', 'com', 'uma', 'um', 'os', 'as', 'se']
     if remover_stopwords:
         palavras = [p for p in palavras if p not in stopwords]
     return ' '.join(palavras)
    
df['texto_pp'] = [preprocessar(t, True) for t in df['texto']]
print(df['texto_pp'].head(3))
from sklearn.model_selection import train_test_split
X = df['texto_pp']
df['texto_keep'] = [preprocessar(t, False) for t in df['texto']]
y = df['sentimento']
idx_treino, idx_teste = train_test_split(df.index, test_size=0.25, stratify=y, random_state=42)
print(len(idx_treino), len(idx_teste))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
configs = []
configs.append(['A', 'mantidas', 'BoW', 'texto_keep', CountVectorizer()])
configs.append(['B', 'mantidas', 'TF-IDF', 'texto_keep', TfidfVectorizer()])
configs.append(['C', 'removidas', 'BoW', 'texto_pp', CountVectorizer()])
configs.append(['D', 'removidas', 'TF-IDF', 'texto_pp', TfidfVectorizer()])
print(len(configs))
linhas = []
for nome, sw, tipo, coluna, vetorizador in configs:
    modelo = Pipeline([('v', vetorizador), ('m', LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Xtr = df.loc[idx_treino, coluna]
    ytr = y.loc[idx_treino]
    acc = cross_val_score(modelo, Xtr, ytr, cv=cv, scoring='accuracy')
    f1 = cross_val_score(modelo, Xtr, ytr, cv=cv, scoring='f1_macro')
    vocab = vetorizador.fit(Xtr).get_feature_names_out().size
    linhas.append([nome, sw, tipo, vocab, acc.mean(), acc.std(), f1.mean(), f1.std()])
print(len(linhas))
comparacao = pd.DataFrame(linhas, columns=['config', 'stopwords', 'vetorizador', 'vocabulario', 'acc_cv_media', 'acc_cv_desvio', 'f1_cv_media', 'f1_cv_desvio'])
print(comparacao)
comparacao.to_csv('resultados/comparacao.csv', index=False)
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
nome, sw, tipo, coluna, vetorizador = configs[0]
modelo_final = Pipeline([('v', vetorizador), ('m', LogisticRegression(max_iter=1000))])
modelo_final.fit(df.loc[idx_treino, coluna], y.loc[idx_treino])
prev = modelo_final.predict(df.loc[idx_teste, coluna])
print(confusion_matrix(y.loc[idx_teste], prev))
print(classification_report(y.loc[idx_teste], prev))
rel = classification_report(y.loc[idx_teste], prev)
with open('resultados/relatorio_classificacao.txt', 'w', encoding='utf-8') as arq:
    print(rel, file=arq)
pd.DataFrame(confusion_matrix(y.loc[idx_teste], prev)).to_csv('resultados/matriz_confusao.csv', index=False)
prob = modelo_final.predict_proba(df.loc[idx_teste, coluna])
i_pos = list(modelo_final['m'].classes_).index('positivo')
erros = df.loc[idx_teste, ['texto']].copy()
erros['real'] = y.loc[idx_teste].values
erros['previsto'] = prev
erros['probabilidade'] = prob[:, i_pos]
erros = erros[erros['real'] != erros['previsto']]
erros.to_csv('resultados/erros.csv', index=False)
print(len(erros))
palavras = modelo_final['v'].get_feature_names_out()
coef = modelo_final['m'].coef_[0]
ordem = np.argsort(coef)
linhas_pal = [['positivo', palavras[i], coef[i]] for i in ordem[-10:][::-1]]
linhas_pal = linhas_pal + [['negativo', palavras[i], coef[i]] for i in ordem[:10]]
pd.DataFrame(linhas_pal, columns=['direcao', 'palavra', 'coeficiente']).to_csv('resultados/palavras_influentes.csv', index=False)
dificeis = pd.read_csv('data/casos_dificeis.csv')
dificeis['texto_pp'] = [preprocessar(t, False) for t in dificeis['texto']]
preds = modelo_final.predict(dificeis['texto_pp'])
prob_d = modelo_final.predict_proba(dificeis['texto_pp'])[:, i_pos]
saida = pd.DataFrame({'texto': dificeis['texto'], 'tipo': dificeis['tipo'], 'esperado': dificeis['sentimento_esperado'], 'previsto': preds, 'prob_positivo': prob_d})
print(saida)
saida.to_csv('resultados/casos_dificeis_predicoes.csv', index=False)
















