# Sistema de Rastreamento de Pêndulo Duplo

## Sobre o Projeto
Este repositório abriga o desenvolvimento de um sistema de rastreamento computacional de alta performance para a análise dinâmica de um **pêndulo duplo físico**, desenvolvido no âmbito de uma **Iniciação Científica (IC)** na Faculdade de Tecnologia da Universidade Estadual de Campinas (FT Unicamp).

O projeto moderniza uma aplicação legada de console em C++, transformando-a em uma interface gráfica moderna, intuitiva e orientada a dados, combinando a flexibilidade do Python com a velocidade de processamento do C++.

## Objetivo
O objetivo principal é capturar, rastrear e registrar com alta precisão a trajetória espacial de duas massas acopladas em um pêndulo duplo utilizando técnicas de visão computacional. Além da física do movimento, o sistema monitora em tempo real o consumo de recursos de hardware (threads do processador) para avaliar a eficiência computacional da arquitetura híbrida implementada.

## Arquitetura e Tecnologias
O sistema adota uma abordagem híbrida otimizada para laboratório:
* **Interface Gráfica (GUI):** Desenvolvida em **Python** utilizando a biblioteca **CustomTkinter**, proporcionando controle de estados (Preview, Contagem Regressiva e Gravação), visualização da câmera e plotagem instantânea de gráficos analíticos.
* **Motor de Processamento (Backend):** Escrito em **C++** para garantir o máximo desempenho no tratamento de imagens.
* **Ponte de Comunicação:** Utiliza a biblioteca **Pybind11** para expor as funções de rastreamento em C++ diretamente como um módulo nativo do Python (`pendulum_tracker`), permitindo a troca de dados em memória sem gargalos.
* **Visão Computacional & OpenCV:** Aplicação de algoritmos de *Template Matching* para a localização robusta das massas do pêndulo em cada quadro de vídeo.
* **Monitoramento de Hardware:** A biblioteca **psutil** coleta métricas dinâmicas do uso de threads da CPU durante o rastreamento.
* **Exportação de Dados:** Os dados brutos de trajetória física e de uso de hardware são salvos automaticamente em arquivos `.csv` estruturados e separados em diretórios específicos (`CSV/pendulo` e `CSV/threads`).

## Funcionalidades do Sistema
1. **Modo Preview com Espelho de Câmera:** Permite o ajuste prévio de enquadramento e iluminação antes de iniciar o experimento.
2. **Contagem Regressiva Automatizada:** Sequência de 3 segundos antes do início da captura para garantir que o pesquisador consiga soltar o pêndulo sem interferências na lente.
3. **Máquina de Estados Visual:** Feedback em tempo real na tela informando se o sistema está em espera, preparando ou gravando.
4. **Análise Pós-Experimento:**
   * Geração de gráficos de trajetória espacial ($X \times Y$) das massas via Matplotlib.
   * Geração de gráficos de uso de threads no tempo ($Tempo \times CPU\%$).
   * Salvamento automático de logs em formato CSV para validações analíticas posteriores.

## Referências e Créditos
Este projeto evoluiu a partir de uma ferramenta acadêmica de rastreamento desenvolvida originalmente para estudos de Pêndulo de Foucault. Expressamos nossos agradecimentos aos autores originais pela base metodológica disponibilizada publicamente:
* **Repositório de Referência:** Repositório original dos colaboradores italianos (documentação e estruturas base legadas).

## Como Executar

### 1. Pré-requisitos
Certifique-se de possuir o compilador C++ (`g++`), CMake e as bibliotecas de desenvolvimento do OpenCV e Pybind11 instaladas no seu ambiente Linux (Ubuntu):
```bash
sudo apt install build-essential cmake libopencv-dev pybind11-dev python3-tk python3-pil.imagetk
```

### 2. Instalação das Dependências Python
```bash
pip3 install customtkinter opencv-python matplotlib psutil pybind11 --break-system-packages
```

### 3. Compilação do Motor C++
Navegue até a pasta de construção e compile o módulo híbrido:
```bash
cd build
rm -rf CMakeCache.txt CMakeFiles/
cmake ..
make
```
*(Certifique-se de copiar o arquivo `.so` gerado para a pasta onde se encontra o script da interface).*

### 4. Execução da Aplicação
```bash
cd release/source
python3 gui.py
```
*(Nota: Pressione **ESC** a qualquer momento para sair do modo de tela cheia).*

---

Desenvolvido por Antonio Carlos Rosendo da Silva — FT Unicamp