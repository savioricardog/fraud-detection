# 🚜 Detecção de Fraudes em transações bancárias (RandomForest + Max_Depth + Class_Weight)

## 📋 Sobre o Projeto
Este projeto resolve um problema muito comum em instituições financeiras: Transações fraudulentas.

O principal desafio foi lidar com dados de **Alta Especificidade** e **Distribuição Assimétrica** (muitas transações normais e pouquissimas de fraudes).

## 🧠 Estratégia de Modelagem

### 1. Algoritmo e Paramêtro
Utilizei o **RandomForest Classifier** com a função objetivo **max_depth** (`None`) e **Clas_Weight** (`None`).
* **Por que Max_Depth e Class_Weight?** Por que no caso de análise de fraudes o mais díficil é entender a especifidades dos padrôes fraudulentos, e neste caso, a melhor solução é não "travar" o
max_depth, permitindo a árvore fazer quantas "perguntas" achar necessário para entender o padrão
da transação fraudulenta, em conjunto com ele, o paramêtro class_weight ajuda muito dizendo para o
modelo dar mais enfoque na classe minoritária (fraude).

### 2. Engenharia de Features
A estrutura de dados foi construída com `Scikit-Learn` incluindo:
* **Escalonamento:** Escalonamento do montante e do tempo para menores escalas.


## 📊 Resultados (Test Validation)

| Métricas | Valor Final |
|----------|-------------|
| **Precision** | **72%** (Assertividade percentual dos apontamentos de fraude) |
| **Recall**    | **97%** (Capacidade de detecção) |
| **F1-Score**  | **83%** (Equilíbrio entre Precision x Recall) |

### Performance: Matrix de Confusão
> *O gráfico de matrix de confusão abaixo mostra como se comportou o modelo durante o teste,
entregando um resultado de apenas **3** fraudes não detectadas*

![Matrix de Confusão](img/confusion_matrix_RF.png)

## 🚀 Como Rodar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [git@github.com:savioricardog/fraud-detection.git](https://github.com/savioricardog/fraud-detection.git)

2. **Instale as dependências:**
   ```bash 
   pip install -r requirements.txt

3. **Execute o arquivos :**
   ```bash 
   python fraud-detection.py

## 📂 Estrutura de Arquivos 

fraud-detection.py: Estrutura principal de treinamento.

fraud-detection.ipynb: Arquivo em modelo Jupyter.

requirements.txt: Dependências do ambiente.

models/model_fraud_V1.pkl: Modelo treinado.


**Desenvolvido por Savio Ricardo Garcia 👨‍💻**
