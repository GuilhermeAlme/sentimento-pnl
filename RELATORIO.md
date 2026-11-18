# RELATORIO — Análise de Sentimento com PLN

**Nome:Guilherme Pinto de Almeida
**Repositório:https://github.com/GuilhermeAlme/sentimento-pnl.git
**Data:18/11/2026 

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420 | 0,856 ± 0,093 | 0,856 ± 0,093 |
| B | mantidas | TF-IDF | 420 | 0,846 ± 0,084 | 0,846 ± 0,084 |
| C | removidas | BoW | 409 | 0,851 ± 0,089 | 0,851 ± 0,089 |
| D | removidas | TF-IDF | 409 | 0,846 ± 0,073 | 0,846 ± 0,073 |

**Configuração escolhida:** A     **Acurácia no teste:** 0,82     **F1 no teste:** 0,81



---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*
Converti tudo para minúsculas antes de qualquer limpeza, porque a expressão regular que usei só reconhece letras minúsculas se ela rodasse primeiro, o trecho em maiúsculo sumiria do texto.
Depois a regex mantém sequências de letras com acento e descarta sozinha pontuação, números e espaços ex: em "tudo estava 2 dias!" sobrou "tudo estava dias".
Usei uma lista de 15 stopwords de português padrão, feita de artigos, preposições, pronomes e conjunções, e não coloquei verbo de sentimento porque é justamente o sinal que diferencia as classes.
O vocabulário aprendido no treino teve 420 palavras com stopwords e 409 sem, então a lista removeu 11 termos.
Mantive a palavra não fora da lista, porque sem ela "A embalagem não é excelente" (linha 4, negativa) viraria "embalagem excelente", e o modelo confirmou essa decisão, dando a não o coeficiente -2,05, o mais negativo de todo o vocabulário.



## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*
BoW conta quantas vezes cada palavra aparece no texto TF-IDF multiplica essa contagem por um fator de raridade no corpus, então palavra que aparece em quase tudo vale menos.
Na CV as médias foram A 0,856, B 0,846, C 0,851, D 0,846, com desvios de 0,093, 0,084, 0,089 e 0,073.
A maior diferença entre elas é 0,010, menor que qualquer desvio padrão, então não é relevante dentro da variação das dobras: as quatro são um empate técnico.
Escolhi a A por ter a melhor média nas duas métricas da validação cruzada, e porque a BoW deixa a leitura dos coeficientes mais direta.
A TF-IDF não rendeu mais aqui porque os textos têm 7,7 palavras em média, quase sem repetição dentro da frase, então não há termo frequente demais para penalizar.


## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*
Matriz de confusão: 25 negativos corretos, 8 reclamações previstas como elogio, 4 elogios previstas como reclamação, 28 positivos corretos. Acurácia 0,82 e F1 macro 0,81.
Negativo ficou com precisão 0,86 e recall 0,76; positivo com precisão 0,78 e recall 0,88.
Para a loja o mais grave é reclamação virar elogio, porque a cliente insatisfeita some sem alerta e sem chance de recuperação o inverso custa só uma oportunidade de marketing.
A acurácia é pouco enganosa porque o dataset é balanceado, 130 e 130; na fraude da Aula 8, quase tudo sendo normal, um modelo que dissesse "não é fraude" sempre teria quase 100% sem servir para nada.


## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*
Cada configuração foi medida em cinco dobras, então eu tenho cinco notas, e não uma a A deu média 0,856 com desvio de 0,093.
As quatro configurações diferem no máximo 0,01 entre si, e o desvio é quase nove vezes maior (0,07 a 0,09). Então numa divisão única, com só 65 frases no teste, quem ganhasse seria sorte de qual frase onde caiu.
Escolher pelo teste estraga o teste ele deixa de ser uma estimativa de como o modelo se comporta em texto nunca visto e vira critério de escolha, e o número final sai inflado. Por isso o teste foi usado uma única vez, depois da escolha feita pela CV.
Vazamento de dados aqui é o vetorizador aprender o vocabulário com os 260 textos, incluindo os 65 de teste, ou aprender fora das dobras durante a CV. Evitei usando Pipeline, que faz o fit do vetorizador dentro de cada dobra, e pegando sempre df.loc[idx_treino, ...] em todo ponto de aprendizado.


## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*
"A série perdeu a graça, enrolação do começo ao fim" virou positivo com 0,56 e "Camiseta encolheu e desbotou depois da primeira lavagem" com 0,59 as palavras que carregam o sentimento (enrolação, encolheu, desbotou) são raras e podem não ter entrado no vocabulário de 420 termos aprendido no treino, então o modelo ficou sem sinal.
"Ficou lindo na sala, combinou perfeitamente com os móveis" (positivo, previsto negativo, 0,48) e "Ficou feio na sala, a cor é bem diferente da foto" (negativo, previsto positivo, 0,55) têm a mesma moldura de frase e rótulos opostos errei os dois nos dois sentidos, porque ele lê palavra isolada, não a frase.
Nenhum dos 12 erros do teste foi convicto todos ficaram entre 0,36 e 0,67 de probabilidade, perto do limite de 0,5.
Tem palavra que não carrega sentimento próprio não (-2,05) é sinal mas não é juízo bastante (+2,34) é intensificador dinheiro (-0,70), mais (-0,63, vem de "nunca mais compro") e toda (-0,68, de "embalagem toda amassada") são vocabulário do domínio e palavras funcionais.
Conclusão o modelo aprendeu correlações deste corpus, não a língua. Ele é bom enquanto a loja escrever como este dataset escreveu.


## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*
Na negação acertou 2 de 4 pegou "Não gostei nada do atendimento" (0,27) e "Não é ruim, gostei bastante" (0,67), mas errou os elogios formulados por negação, "Não tenho nada a reclamar, tudo perfeito" (0,44) e "Sem defeitos, funciona direitinho" (0,44).
Na ironia acertou 2 de 3 e a que errou veio convicta "Que ótimo, o produto chegou quebrado" com 0,83, porque ótimo é muito forte no lado positivo.
As duas frases mistas saíram positivo (0,79 e 0,88) e "Chegou ontem" deu 0,5066, quase o empate, mostrando que ele não tinha evidência; como o modelo só conhece positivo e negativo, misto e neutro não são comparáveis.
Isso vem de BoW e TF-IDF perderem a ordem não e bom são colunas separadas e nada compõe a negação. Melhorias: n-gramas (1,2) para "não gostei" virar um sinal só, e embeddings com contexto para ordem e ironia.
Limitações do dataset: parte gerada por padrões, então os números são otimistas, e são só 260 frases de 7,7 palavras. Risco em produção: quando a linguagem muda, o modelo cai em silêncio, e as falhas que passam despercebidas são justamente ironia e reclamação lida como elogio.



---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [ x] Usei IA generativa **somente** para: Usei IA generativa somente para esclarecer conceitos de PLN e de métricas, revisar mensagens de erro e conferir os números, além da lista de stopwords o código e o texto deste relatório foram escritos por mim.(não é permitido gerar código ou relatório).