import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

# Script para testar o modelo Fine-Tuned (LoRA) isoladamente
# Este script carrega o modelo base e aplica os pesos que você treinou

def testar_modelo_lora(adapter_path):
    print(f"=== Iniciando Teste do Modelo LoRA ===")
    print(f"Procurando adaptador em: {os.path.abspath(adapter_path)}")
    
    try:
        config = PeftConfig.from_pretrained(adapter_path)
    except Exception as e:
        print(f"\nERRO CRÍTICO: Não foi possível carregar a configuração em '{adapter_path}'.")
        print(f"Detalhes: {e}")
        print("Verifique se o caminho está correto e se a pasta contém o arquivo 'adapter_config.json'")
        return

    base_model_name = config.base_model_name_or_path
    print(f"Modelo base identificado na config: {base_model_name}")

    # Configuração de dispositivo (GPU se disponível, senão CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Carrega o Tokenizer e o Modelo Base
    print("Carregando modelo base e tokenizer (isso pode demorar um pouco)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        load_in_8bit=True if device == "cuda" else False, # Adicionado para carregar em 8 bits se GPU
        device_map="auto", # Sempre usar device_map="auto" para carregar o modelo corretamente
        quantization_config=None, # Desativar quantização para o modelo base
        trust_remote_code=True,
        use_safetensors=True
    )

    # Carrega e acopla o adaptador LoRA ao modelo base
    print(f"Aplicando adaptador LoRA...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("\nModelo carregado com sucesso! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Digite sua pergunta (ex: Sintomas de sepse): ")
        if pergunta.lower() in ['sair', 'exit']:
            break
            
        # 1. Formatação do Prompt: CRUCIAL para o modelo entender que deve responder
        prompt_formatado = f"### Instrução:\n{pergunta}\n\n### Resposta:\n"
        
        inputs = tokenizer(prompt_formatado, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=150,           # Limite o tamanho da resposta
                do_sample=False,              # Desativa a amostragem
                temperature=0.0,              # Totalmente determinístico (foco em fatos)
                # top_p=0.9,                  # Desnecessário quando do_sample=False
                repetition_penalty=1.2,       # Evita loops e repetições
                pad_token_id=tokenizer.eos_token_id # Evita erros de padding
            )
            
        resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove o prompt da resposta para exibir apenas o texto gerado
        resposta_gerada = resposta_completa.replace(prompt_formatado, "").strip()
        
        print(f"\n--- Resposta do Modelo ---\n{resposta_gerada}\n--------------------------\n")

if __name__ == "__main__":
    # IMPORTANTE: Ajuste este caminho para onde a pasta 'med-assistant-lora' foi gerada
    # O código abaixo assume uma estrutura de projeto específica.
    
    # Obtém o diretório do script atual (src/)
    script_dir = Path(__file__).resolve().parent
    # Sobe para a pasta Tech Challenge 3 (ajuste conforme sua estrutura)
    project_dir = script_dir.parent
    # Caminho para o modelo
    model_path = project_dir / "fine_tuned_model" / "med-assistant-lora"
    
    if model_path.exists():
        testar_modelo_lora(str(model_path))
    else:
        print(f"ERRO: Caminho não encontrado: {model_path}")
        print(f"Verifique se a pasta 'med-assistant-lora' existe em: {project_dir / 'fine_tuned_model'}")
