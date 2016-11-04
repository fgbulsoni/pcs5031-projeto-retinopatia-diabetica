# pcs5031-projeto-retinopatia-diabetica para Introdução a Ciência dos Dados


Projeto para disciplina de Introdução à Ciência dos Dados (PCS5031) na Escola Politécnica da Universidade de São Paulo (Poli-USP).

#### Índice
1. [Introdução](#introdução)
2. [Papéis e Responsabilidades](#papéis-e-Responsabilidades)
3. [Metodologia](#metodologia)
4. [Utilização](#utilização)
5. [Autores](#autores)


## Introdução

O objetivo desse projeto é a aplicação das técnicas analíticas e de gerência de dados discutidas ao longo da matéria.

Técnicas de seleção de dados, amostragem, normalização e análise serão aplicadas em um banco de imagens com o intuito da criação de um modelo capaz de detecção de Retinopatia Diabética.


## Papéis e Responsabilidades

Durante a fase de concepção desse projeto um Data Management Plan (DMP) foi gerado, o mesmo contém informações detalhadas sobre os planos para os dados gerados ou analisados nesse projeto e pode ser acessado [aqui](DMP-pcs5031-projeto-retinopatia-diabetica.pdf).

O banco de imagens utilizado está disponível em domínio público e está sob responsabilidade da plataforma [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).

Para extração de atributos a partir das imagens, funções da plataforma Adessowiki da Universidade Estadual de Campinas foram utilizadas. Tais funções são parte de código proprietário e a instituição deverá ser contatada caso exista necessidade de acesso ao mesmo.

Todos os códigos fonte gerados para a análise dos atributos extraídos das imagens estão disponibilizados neste repositório sob licença open source [Apache 2.0](LICENSE). Os atributos extraídos e outros dados relevantes gerados pelo processamento ou ao longo do projeto estão armazenados no repositório [DataONE](LINK PARA REPOSITORIO).

## Metodologia

Os dados gerados nesta pesquisa são os atributos extraídos de um conjunto de imagens de domínio público.
Primeiramente, as imagens foram redimensionadas e colocadas em escala de cinza.

Então, informações relacionadas ao histograma, comprimento de corrida, e matriz de concorrência foram extraídas. Estes atributos estão disponibilizados em formato .dat e CSV na plataforma [DataONE](LINK PARA REPOSITORIO).

Com os dados em formato CSV, foram aplicados métodos de aprendizado de máquina para verificar se esses atributos eram suficientes para classificar as imagens em saudáveis/doentes e assim foi gerado um modelo.

## Utilização

Para conversão dos dados .dat para .csv, utilize o a função a seguir disponível na library [algo.py](lib/algo.py):

```python
Some code snippet here
```

Depois da conversão para .csv, execute o script [algo2.py](lib/algo2.py) da seguinte forma para a análise dos dados:
```python
Some code snippet here
```

## Autores
- Felipe Garcia Bulsoni - [@fgbulsoni](https://github.com/fgbulsoni)
- Jose Augusto Salim - [@zedomel](https://github.com/zedomel)
- Thamires de Campos Luz - [@thamiluz](https://github.com/thamiluz)
