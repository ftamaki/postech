# fine_tunning.py
import json
import traceback
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATASET_PATH = PROJECT_ROOT / "data" / "fine_tuning_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "trained_data" / "fine_tuned_model_lora"

MODEL_NAME = "facebook/opt-350m"
NEW_MODEL_NAME = "med-assistant-lora"


# ==============================================================================
# 2. QUANTIZAÇÃO (GPU)
# ==============================================================================

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)


# ==============================================================================
# 3. LORA
# ==============================================================================

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


# ==============================================================================
# 4. TRAINING ARGUMENTS
# ==============================================================================

training_arguments = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    #learning_rate=2e-4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    fp16=False, # <-- MUDANÇA FINAL: Desativar completamente o AMP do Trainer
    bf16=False, # Manter para garantir
    optim="adamw_torch",
    report_to="none",
)

# ==============================================================================
# 5. FORMATTING FUNCTION
# ==============================================================================

def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output}"
    )


# ==============================================================================
# 6. FINE-TUNING
# ==============================================================================

def fine_tune_model():
    print(f"\n📌 Carregando dataset: {DATASET_PATH}")

    try:
        data = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    except Exception:
        dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

    if len(dataset) == 0:
        raise RuntimeError("Dataset vazio")

    print(f"✔ Dataset carregado com {len(dataset)} exemplos")

    print(f"\n📌 Carregando modelo base: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16,
        use_safetensors=True
        #device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("✔ Modelo e tokenizer carregados")

    print("\n📌 Inicializando SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_example,
    )

    print("✔ Trainer inicializado")

    print("\n🚀 Iniciando treinamento...")
    trainer.train()
    print("✔ Treinamento concluído")

    print("\n💾 Salvando modelo...")
    save_path = OUTPUT_DIR / NEW_MODEL_NAME
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✔ Modelo salvo em: {save_path}")


# ==============================================================================
# 7. MAIN
# ==============================================================================

if __name__ == "__main__":

    print("\n================ VERSÕES =================")
    import transformers, trl, peft, accelerate, bitsandbytes, datasets

    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("trl:", trl.__version__)
    print("peft:", peft.__version__)
    print("accelerate:", accelerate.__version__)
    print("bitsandbytes:", bitsandbytes.__version__)
    print("datasets:", datasets.__version__)

    print("\n================ GPU =================")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("❌ CUDA NÃO disponível")

    print("\n================ TREINAMENTO =================")

    try:
        fine_tune_model()
        print("\n✅ FINE-TUNING CONCLUÍDO COM SUCESSO")
    except Exception as e:
        print("\n❌ ERRO NO TREINAMENTO")
        print(str(e))
        print(traceback.format_exc())
