# fine_tunning.py
# Resumo do script
#   -Esse script realiza o fine-tuning de um modelo de linguagem usando LoRA e TRL.
# Fine tunning significa ajustar um modelo pré-treinado em um conjunto de dados específico para melhorar seu desempenho em uma tarefa particular.
#   - Nesse caso, o script utiliza um dataset JSONL para treinar o modelo o fine_tunning_dataset.jsonl localizado na pasta data.
#   - Fine tuning é uma técnica comum em aprendizado de máquina, especialmente em modelos de linguagem natural.
# Este script realiza o fine-tuning de um modelo de linguagem usando LoRA (Low-Rank Adaptation) e TRL (Transformers Reinforcement Learning).
# Requer bibliotecas: transformers, peft, trl, datasets, torch.
# Configurações de modelo, LoRA e argumentos de treinamento são definidos.
# O script carrega o dataset, inicializa o modelo e o treinador, e executa o fine-tuning.
import os
from pathlib import Path                    # Manipulação de caminhos de arquivos
from datasets import load_dataset, Dataset  # Carregamento e manipulação de datasets
from transformers import (                  
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)                                           # AutoModelForCausalLM: Carrega modelos de linguagem causal pré-treinados
                                            # AutoTokenizer: Tokenizador para modelos de linguagem
                                            # BitsAndBytesConfig: Configuração para quantização de modelos
                                            # TrainingArguments: Argumentos para treinamento de modelos
from peft import LoraConfig                 # Configuração do LoRA para fine-tuning eficiente, Low-Rank Adaptation 
from trl import SFTTrainer                  # SFTTrainer: Treinador especializado para fine-tuning supervisionado de modelos de linguagem
import torch                                # Biblioteca principal para computação em tensores e aprendizado de máquina
# através do torch que realizamos operações em GPU/CPU, manipulação de tensores, construção e treinamento de modelos neurais
# nesse script usamos torch para definir o dtype (float16) e device_map (auto) ao carregar o modelo, permitindo otimização de desempenho durante o fine-tuning
import json                                 # Manipulação de dados JSON   
import traceback                            # Tratamento e exibição de rastreamentos de erros


# Script dividido em 5 partes principais:
# 1. Configurações - Caminhos e nomes de modelos
# 2. Configuração do LoRA - Parâmetros do LoRA para fine-tuning
# 3. Argumentos de Treinamento - Configurações para o processo de treinamento
# 4. Função de Fine-Tuning - Carrega dados, modelo e executa o fine-tuning
# 5. Main + Debug - Execução do fine-tuning com tratamento de erros
# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()               # Raiz do projeto (dois níveis acima do arquivo atual)
DATASET_PATH = PROJECT_ROOT / "data" / "fine_tuning_dataset.jsonl"  # Caminho para o dataset de fine-tuning
OUTPUT_DIR = PROJECT_ROOT / "fine_tuned_model"                      # Diretório para salvar o modelo fine-tuned

model_name = "facebook/opt-125m"    # Modelo base pré-treinado, pode ser alterado para outro modelo compatível, escolhido por ser leve para testes
new_model = "med-assistant-lora"    # Nome do novo modelo fine-tuned que será salvo


# ==============================================================================
# 2. LORA
# Low-Rank Adaptation (LoRA) é uma técnica de fine-tuning eficiente que adapta modelos pré-treinados
# reduzindo o número de parâmetros treináveis, permitindo ajustes rápidos com menos dados computacionais.
# LoRA insere matrizes de baixa-rank em camadas específicas do modelo, facilitando o aprendizado de novas tarefas sem modificar os pesos originais do modelo.
# ==============================================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)   # Configuração do LoRA para fine-tuning eficiente
    # r: rank das matrizes LoRA - determina a capacidade de adaptação do modelo
    # lora_alpha: fator de escala para LoRA - controla a influência das adaptações LoRA
    # lora_dropout: taxa de dropout para regularização - ajuda a prevenir overfitting
    # bias: tratamento de bias (nenhum nesse caso) - como lidar com termos de bias nas camadas adaptadas
    # task_type: tipo de tarefa (modelos de linguagem causal) - especifica a natureza da tarefa de fine-tuning
    # target_modules: módulos do modelo onde LoRA será aplicado (projeções Q e V) - define quais partes do modelo serão adaptadas usando LoRA


# ==============================================================================
# 3. TRAINING ARGUMENTS
# Configurações para o processo de treinamento do modelo
# ==============================================================================

training_arguments = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    bf16=False,
    optim="adamw_torch",
)   # Argumentos de treinamento para o fine-tuning
    # output_dir: diretório para salvar checkpoints e o modelo final
    # num_train_epochs: número de épocas de treinamento - quantas vezes o modelo verá todo o dataset durante o treinamento
    # per_device_train_batch_size: tamanho do batch por dispositivo (GPU/CPU) - quantos exemplos são processados antes de atualizar os pesos do modelo
    # gradient_accumulation_steps: passos de acumulação de gradiente para simular batches maiores - permite efetivamente aumentar o tamanho do batch sem aumentar o uso de memória
    # learning_rate: taxa de aprendizado para o otimizador - controla a velocidade com que o modelo ajusta seus pesos durante o treinamento
    # logging_steps: frequência de logging durante o treinamento - quantas vezes os logs de treinamento são registrados
    # save_steps: frequência de salvamento do modelo durante o treinamento - quantas vezes o modelo é salvo durante o treinamento
    # fp16: se usar precisão de ponto flutuante 16 (não usado)
    # bf16: se usar bfloat16 (não usado)
    # optim: otimizador a ser usado (AdamW implementado no PyTorch) - especifica o algoritmo de otimização para atualizar os pesos do modelo durante o treinamento


# ==============================================================================
# 4. FINE-TUNING
# Função principal que realiza o fine-tuning do modelo
# ==============================================================================

def fine_tune_model():

    print(f"\n📌 Carregando dataset de: {DATASET_PATH}") # Carrega o dataset JSONL para fine-tuning

    try:
        data_list = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line))      # Carrega manualmente o dataset JSONL linha por linha

        dataset = Dataset.from_list(data_list)          # Converte a lista de dicionários em um Dataset do Hugging Face, cada linha do arquivo JSONL é um exemplo separado no dataset      

    except Exception as e:
        print(f"Falhou ao carregar manualmente. Usando método HF. Erro: {e}")
        dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

    if len(dataset) == 0:
        raise Exception("Dataset vazio! Verifique o arquivo JSONL.")

    print(f"✔ Dataset carregado com {len(dataset)} exemplos")

    print(f"\n📌 Carregando modelo base: {model_name}")

    # Configuração para quantização do modelo usando 8 bits, reduzindo o uso de memória

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,   
    )   # Carrega o modelo pré-treinado com configuração para uso em GPU/CPU e dtype float16
    # aqui usamos torch.float16 para reduzir o uso de memória e acelerar o treinamento, especialmente em GPUs compatíveis

    # tokenizer utiliza AutoTokenizer para garantir compatibilidade com o modelo pré-treinado, aqui ele é carregado com trust_remote_code=True para permitir o uso de código personalizado hospedado remotamente, se necessário
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    # Carrega o tokenizador correspondente ao modelo pré-treinado,
    tokenizer.pad_token = tokenizer.eos_token # Define o token de padding como o token de fim de sequência (eos_token)
    # Isso é importante para garantir que o tokenizador possa lidar corretamente com sequências de diferentes comprimentos durante o treinamento e inferência

    print("✔ Modelo carregado")

    # ==============================================================================
    # TRL 0.25.1 SFT TRAINER - Fine-Tuning Supervisionado
    # Usamos o SFTTrainer do TRL para realizar o fine-tuning supervisionado do modelo
    # ==============================================================================

    print("\n📌 Inicializando SFTTrainer...")

    # Inicializa o treinador SFT com o modelo, argumentos de treinamento, dataset e configuração do LoRA
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
    )   
    # STFTrainer é especializado para fine-tuning supervisionado de modelos de linguagem, facilitando a integração com LoRA e gerenciamento de datasets
    # model: modelo pré-treinado a ser fine-tuned
    # args: argumentos de treinamento definidos anteriormente
    # train_dataset: dataset de treinamento carregado
    # eval_dataset: dataset de avaliação (nenhum nesse caso)
    # peft_config: configuração do LoRA para adaptação eficiente do modelo
    # essa configuração permite que o treinador gerencie o processo de fine-tuning de forma eficiente, aplicando as adaptações LoRA conforme especificado
    # isso simplifica o processo de treinamento, especialmente para grandes modelos de linguagem
 

    print("✔ Trainer inicializado")

    # ==============================================================================
    # TREINAMENTO
    # Executa o fine-tuning do modelo usando o treinador configurado
    # execução simples mas eficaz do processo de treinamento
    # ==============================================================================

    print("\n🚀 Iniciando treinamento...")
    trainer.train() # Inicia o processo de fine-tuning do modelo
    print("✔ Treinamento concluído")

    print("\n💾 Salvando modelo fine-tuned...")
    save_path = OUTPUT_DIR / new_model
    trainer.model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print(f"✔ Modelo salvo em: {save_path}")


# ==============================================================================
# 5. MAIN + DEBUG
# Ponto de entrada do script com tratamento de erros para facilitar o debug
# ==============================================================================

if __name__ == "__main__": # Ponto de entrada do script __main__ indica que o código dentro desse bloco será executado apenas quando o script for executado diretamente, não quando importado como módulo

    print("\n====================================================")
    print("📦 VERSÕES DO AMBIENTE")
    print("====================================================")
    print(f"torch: {torch.__version__}")                                # Versão do PyTorch, biblioteca principal para computação em tensores e aprendizado de máquina
    import transformers, trl, peft, accelerate, bitsandbytes, datasets  # Importa bibliotecas para exibir suas versões
    print(f"transformers: {transformers.__version__}")                  # Versão da biblioteca Transformers, usada para modelos de linguagem pré-treinados  
    print(f"trl: {trl.__version__}")                                    # Versão da biblioteca TRL (Transformers Reinforcement Learning), usada para fine-tuning de modelos de linguagem
    print(f"peft: {peft.__version__}")                                  # Versão da biblioteca PEFT (Parameter-Efficient Fine-Tuning), usada para fine-tuning eficiente com LoRA
    print(f"accelerate: {accelerate.__version__}")                      # Versão da biblioteca Accelerate, usada para facilitar o treinamento em múltiplos dispositivos
    print(f"bitsandbytes: {bitsandbytes.__version__}")                  # Versão da biblioteca BitsAndBytes, usada para quantização de modelos
    print(f"datasets: {datasets.__version__}")                          # Versão da biblioteca Datasets, usada para carregamento e manipulação de datasets
    print("====================================================\n")

    # GPU CHECK
    print("🔍 STATUS GPU:")
    if torch.cuda.is_available():                                        # Verifica se uma GPU CUDA está disponível para uso
        print(f" - GPU detectada: {torch.cuda.get_device_name(0)}")      # Exibe o nome da GPU detectada
    else:
        print(" - Sem GPU CUDA detectada (treino será no CPU — lento!)") # Aviso se nenhuma GPU for detectada, indicando que o treinamento será realizado na CPU, o que é mais lento

    print("\n====================================================")
    print("🏁 EXECUTANDO FINE-TUNING")
    print("====================================================")

    try:
        fine_tune_model()   # Chama a função principal de fine-tuning, nesse caso dentro de um bloco try-except para capturar erros
        print("\n✅ FINE-TUNING CONCLUÍDO COM SUCESSO")

    except Exception as e:
        print("\n❌ ERRO NO TREINAMENTO")
        print("Mensagem do erro:", str(e))
        print("\n🔍 Traceback completo:")
        print(traceback.format_exc())
        print("\n🔧 POSSÍVEIS CAUSAS:")
        print("1. Versão incompatível do TRL, Transformers ou PEFT.")
        print("2. Campo 'text' ausente no JSONL.")
        print("3. Modelo muito grande para CPU.")
        print("4. BitsAndBytes tentando rodar em CPU.")
        print("====================================================\n")
