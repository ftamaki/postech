import os
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import torch
import json
import traceback


# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATASET_PATH = PROJECT_ROOT / "data" / "fine_tuning_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "fine_tuned_model"

model_name = "facebook/opt-125m"
new_model = "med-assistant-lora"


# ==============================================================================
# 2. LORA
# ==============================================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


# ==============================================================================
# 3. TRAINING ARGUMENTS
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
)


# ==============================================================================
# 4. FINE-TUNING
# ==============================================================================

def fine_tune_model():

    print(f"\n📌 Carregando dataset de: {DATASET_PATH}")

    try:
        data_list = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line))

        dataset = Dataset.from_list(data_list)

    except Exception as e:
        print(f"Falhou ao carregar manualmente. Usando método HF. Erro: {e}")
        dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

    if len(dataset) == 0:
        raise Exception("Dataset vazio! Verifique o arquivo JSONL.")

    print(f"✔ Dataset carregado com {len(dataset)} exemplos")

    print(f"\n📌 Carregando modelo base: {model_name}")

    # BitsAndBytes não funciona em CPU — só carregar sem quantização
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,   # ← SOLUÇÃO
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("✔ Modelo carregado")

    # ==============================================================================
    # TRL 0.25.1 — ATENÇÃO: API NOVA
    # ==============================================================================

    print("\n📌 Inicializando SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
    )

    print("✔ Trainer inicializado")

    # ==============================================================================
    # TREINAMENTO
    # ==============================================================================

    print("\n🚀 Iniciando treinamento...")
    trainer.train()

    print("\n💾 Salvando modelo fine-tuned...")
    save_path = OUTPUT_DIR / new_model
    trainer.model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print(f"✔ Modelo salvo em: {save_path}")


# ==============================================================================
# 5. MAIN + DEBUG
# ==============================================================================

if __name__ == "__main__":

    print("\n====================================================")
    print("📦 VERSÕES DO AMBIENTE")
    print("====================================================")
    print(f"torch: {torch.__version__}")
    import transformers, trl, peft, accelerate, bitsandbytes, datasets
    print(f"transformers: {transformers.__version__}")
    print(f"trl: {trl.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")
    print(f"datasets: {datasets.__version__}")
    print("====================================================\n")

    # GPU CHECK
    print("🔍 STATUS GPU:")
    if torch.cuda.is_available():
        print(f" - GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print(" - Sem GPU CUDA detectada (treino será no CPU — lento!)")

    print("\n====================================================")
    print("🏁 EXECUTANDO FINE-TUNING")
    print("====================================================")

    try:
        fine_tune_model()

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
