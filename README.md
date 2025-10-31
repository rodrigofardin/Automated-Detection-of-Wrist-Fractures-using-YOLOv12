# 🦴 Automated Detection of Wrist Fractures using YOLOv12

> Fine-tuning de modelos **YOLOv12** para detecção automática de fraturas em imagens de raio-X pediátricas do dataset **GRAZPEDWRI-DX**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv12](https://img.shields.io/badge/Model-YOLOv12-black)](https://github.com/sunsmarterjie/yolov12)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-API-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Sobre o projeto

Fraturas da mão e do punho, especialmente do rádio e da ulna distal, são comuns em crianças e adolescentes e exigem diagnóstico rápido e preciso.  
Embora a radiografia digital seja amplamente utilizada, fraturas sutis podem passar despercebidas, aumentando o risco de erros diagnósticos e impactos à segurança do paciente.  

Este projeto aplica **Técnicas de Aprendizado Profundo**, em especial **Redes Neurais Convolucionais da família YOLO (You Only Look Once)**, para a **detecção automática de fraturas** em imagens do dataset **GRAZPEDWRI-DX**.  
O modelo **YOLOv12-L** alcançou resultados competitivos e eficiência computacional, demonstrando potencial para integração em sistemas de **diagnóstico assistido por computador**.

---

## 📁 Estrutura do repositório

- `fine-tuning-yolov12.ipynb`: notebook com o pipeline de fine‑tuning (pré-processamento, treino com a API Ultralytics/YOLO, validação e plots de resultados). Este é o código principal que reproduz o experimento.
- `split_check.py` e `split_check_usage.ipynb`: utilitários para dividir o dataset por paciente, checar vazamentos e organizar o dataset no formato YOLO.
- `README.md`: este arquivo — instruções para reproduzir o experimento.
- `requirements.txt`: dependências necessárias.


---

## ⚙️ Instalação e ambiente (Windows - PowerShell)

Crie um ambiente virtual e instale as dependências:

```powershell
# 1) Criar e ativar ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Atualizar instaladores básicos
python -m pip install --upgrade pip setuptools wheel

# 3) Instalar PyTorch com CUDA (exemplo para CUDA 12.4)
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124 torchvision --extra-index-url https://pypi.org/simple

# 4) Instalar demais dependências (inclui YOLOv12 via Git)
pip install -r requirements.txt

```

### Observações:

- A linha `git+https://github.com/sunsmarterjie/yolov12` em `requirements.txt` instalará a versão oficial do repositório [YOLOv12](https://github.com/sunsmarterjie/yolov12), proposto por Tian *et al.* (2025):

### Verificação rápida

Após a instalação verifique as versões e se o CUDA está disponível:

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
python -c "import ultralytics; print('ultralytics', getattr(ultralytics, '__version__', 'git'))"
```

## 📦 Download do dataset

Baixe o dataset [**GRAZPEDWRI-DX**](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) no site oficial e coloque os dados no diretório do projeto com a seguinte estrutura mínima:

```
dataset.csv
images/    # imagens originais (.png)
labels/    # labels YOLO (.txt)
meta.yaml
```

## 🧩 Divisão dos dados (pré‑processamento)

A divisão por paciente (para evitar data leakage) é feita com `split_check.py` e salva em `./splits`:

```powershell
python split_check.py --csv dataset.csv --out splits --patient_col patient_id
```

Ou abra e execute `split_check_usage.ipynb` (ele reproduz as mesmas etapas). Os arquivos gerados serão `train.csv`, `val.csv`, `test.csv` e listas de pacientes por split.

Observação: a divisão usada no artigo será disponibilizada em `./splits` ou por link público quando o artigo for publicado.

### 🧾 Preparar `data.yaml` para treino

Use o utilitário `split_check.organize_yolo(...)` ou a célula correspondente no notebook para copiar imagens/labels para `dataset_yolo/{train,val,test}/{images,labels}` e gerar um `data.yaml`. O utilitário também tenta copiar/ajustar um `meta.yaml` original (removendo placeholders `FILL IN`). Não armazene caminhos absolutos no `data.yaml` do repositório — prefira caminhos relativos.

```
dataset_yolo/
 ├── train/
 │   ├── images/
 │   └── labels/
 ├── val/
 │   ├── images/
 │   └── labels/
 └── test/
     ├── images/
     └── labels/
```

## 🚀 Reproduzindo o experimento (treinamento)

Este repositório disponibiliza o pipeline de treino como notebook interativo — `fine-tuning-yolov12.ipynb` — que é a fonte canônica para reproduzir o experimento.

- Abra `fine-tuning-yolov12.ipynb` no Jupyter/VS Code, ajuste as células indicadas (paths, device, epochs, batch) e execute as células na ordem. O notebook contém o código usado no experimento (chamada a `model.train(...)` via Ultralytics API), geração de métricas e plots.

### Resultados e logs

Os resultados do treino (pesos, figuras e métricas) são salvos na pasta criada pelo Ultralytics (ex: `runs/train/<experiment_name>`). No notebook, gráficos PR/curves são gerados e salvos automaticamente.

Tabela de performance (exemplo)

| Model | Test Size |  Param. | FLOPs | F1 Score | AP50val | AP50-95val | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv12 L | 640 | 26.4M | 82.1G | 67.7% | 66.1% | 42.9% | 24.3ms |

(Substitua pela tabela final do seu artigo — coloque os números reais do seu experimento aqui.)

Citation

Se este trabalho for útil para sua pesquisa, considere citar (exemplo):

```
@misc{fardin2025wristfractures,
  title        = {Automated Detection of Wrist Fractures using YOLOv12},
  author       = {Fardin, Rodrigo},
  year         = {2025},
  howpublished = {\url{https://github.com/rodrigofardin/Automated-Detection-of-Wrist-Fractures-using-YOLOv12}},
  note         = {Fine-tuning de modelos YOLOv12 no dataset GRAZPEDWRI-DX para detecção de fraturas ósseas pediátricas.}
}
```

## 🤝 Contribuição

Contribuições são bem-vindas!
Algumas melhorias planejadas incluem:

-  Implementar benchmarks adicionais (YOLOv13, RF-DERT)
-  Adicionar testes automatizados e checagem de integridade das imagens

