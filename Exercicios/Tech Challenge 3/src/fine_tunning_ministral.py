import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
from pathlib import Path

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

# Nomes dos modelos e arquivos
# Diretório base do arquivo atual (fine_tunning.py)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

base_model = "mistralai/Mistral-7B-v0.1"
new_model = "med-assistant-mistral-lora"
dataset_file = PROJECT_ROOT / "data" / "fine_tuning_dataset.jsonl"
dataset_file = dataset_file.resolve()
OUTPUT_DIR = PROJECT_ROOT / "trained_data" / "fine_tuned_model_lora"
NEW_MODEL_NAME = "med-assistant-mistral-lora"

# Hiperparâmetros LoRA (Agressivos para dataset pequeno)
lora_r = 32
lora_alpha = 64
lora_dropout = 0.1
# Módulos alvo para adaptação (Mistral-7B)
lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Hiperparâmetros de Treinamento
num_train_epochs = 2 # Aumentado para forçar a memorização do dataset de 200 linhas
learning_rate = 2e-5 # Taxa de aprendizado ligeiramente maior
batch_size = 4 # Ajuste conforme sua VRAM
gradient_accumulation_steps = 1 # Ajuste conforme sua VRAM
warmup_ratio = 0.03
logging_steps = 25
save_steps = 50
max_seq_length = 512 # Tamanho máximo da sequência (ajuste se suas respostas forem muito longas)

# ==============================================================================
# 2. FUNÇÃO DE CARREGAMENTO E TREINAMENTO
# ==============================================================================

def train_mistral_qlora():
    # 2.1. Configuração QLoRA (Quantização)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 2.2. Carregar o Modelo Base (Mistral-7B)
    print(f"Carregando modelo base: {base_model} com QLoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepara o modelo para treinamento k-bit (QLoRA)
    model = prepare_model_for_kbit_training(model)

    # 2.3. Carregar o Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Configurações do tokenizer para treinamento
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Necessário para modelos causais

    # 2.4. Carregar o Dataset
    print(f"Carregando dataset de fine-tuning: {dataset_file}")
    # O formato 'json' é usado para arquivos .jsonl
    try:
        dataset = load_dataset("json", data_files=str(dataset_file), split="train")
    except FileNotFoundError:
        print(f"\nERRO: Arquivo de dataset '{dataset_file}' não encontrado.")
        print("Certifique-se de que o arquivo JSONL está no mesmo diretório.")
        return

    # 2.5. Configuração LoRA
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    # 2.6. Configuração de Treinamento
    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit", # Otimizador otimizado para QLoRA
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False, # Usar bfloat16 do QLoRA
        bf16=True, # Habilitar bfloat16 (se a GPU suportar, como a RTX 4070)
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none", # Desativa relatórios externos
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    # 2.7. Inicializar o SFT Trainer
# Tokenização
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
        processing_class=tokenizer,
    )

    # 2.8. Treinar o Modelo
    print("\nIniciando o treinamento...")
    trainer.train()

    # 2.9. Salvar o Adaptador LoRA
    save_path = OUTPUT_DIR / NEW_MODEL_NAME
    trainer.model.save_pretrained(save_path)
    print(f"\nTreinamento concluído. Adaptador LoRA salvo em: {new_model}")

    # 2.10. Opcional: Mesclar o adaptador com o modelo base (para inferência mais fácil)
    # Isso requer VRAM suficiente para carregar o modelo base em FP16 (~14.4 GB)
    # Se você não tiver VRAM suficiente, pule esta etapa e use o código de inferência LoRA.
    # print("Mesclando adaptador LoRA com o modelo base...")
    # model.config.use_cache = True
    # final_model = PeftModel.from_pretrained(model, new_model)
    # final_model = final_model.merge_and_unload()
    # final_model.save_pretrained(f"{new_model}-merged")
    # tokenizer.save_pretrained(f"{new_model}-merged")
    # print(f"Modelo mesclado salvo em: {new_model}-merged")

if __name__ == "__main__":
    train_mistral_qlora()
