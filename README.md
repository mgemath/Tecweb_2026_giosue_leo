# Algoritmo KNN

O algoritmo **K-NN (K-Nearest Neighbors)** é provavelmente o algoritmo de *Machine Learning* mais antigo. A área que hoje chamamos de Ciência de Dados começou com este tipo de abordagem nos anos 1950, com os trabalhos de **Fix e Hodges**.  
Mesmo hoje, continua sendo um dos algoritmos mais importantes, utilizado em diversas aplicações do mundo real, como:

- Sistemas de recomendação  
- Reconhecimento de imagens  
- Detecção de fraude e anomalias  
- Diagnóstico em saúde e análise de similaridade entre pacientes  
- Classificação de texto  
- Reconhecimento de padrões  
- Serviços baseados em localização (lugares mais próximos)  
- Regressão  
- Agregação de grafos em *Graph Neural Networks*  

O algoritmo é conceitualmente simples:  
ele observa um ponto e infere sua classe olhando para seus vizinhos mais próximos, escolhendo a classe que recebe a **maioria dos votos**.

---

## Algoritmo KNN em Julia

O script [`mnist_knn.jl`](mnist_knn.jl) é uma demonstração simples do uso de **Julia** como linguagem para Ciência de Dados, aplicando o algoritmo K-NN ao conjunto de dados **MNIST**, com parte dos rótulos corrompidos artificialmente.

A busca de vizinhos é feita por força bruta usando uma **BruteTree**, ou seja, calculando todas as distâncias diretamente, sem particionamento espacial.  
Surpreendentemente, essa abordagem pode ser mais eficiente do que estruturas de particionamento quando a dimensionalidade dos dados é muito alta.

A escolha do conjunto de dados MNIST foi feita porque ele é grande o suficiente para observar o efeito do parâmetro **k** na precisão do modelo.  
Inicialmente, o desempenho melhora com o aumento de **k**, atingindo um valor máximo, e depois começa a degradar.

Um dos principais desafios do K-NN é que o valor ótimo de **k** precisa ser ajustado por busca experimental, e também depende do tamanho do conjunto de dados.  
À medida que o conjunto de dados cresce, encontrar o valor ótimo de **k** torna-se computacionalmente mais caro.

Esse efeito — o deslocamento do melhor valor de **k** conforme o tamanho do conjunto de dados — pode ser observado no script alterando os parâmetros:

```julia
LIMIT_TRAIN = 10_000
LIMIT_TEST  = 2_000
