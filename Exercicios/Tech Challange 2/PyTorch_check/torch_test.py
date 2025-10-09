import torch

# 1. Verifica se o PyTorch foi compilado com suporte a CUDA
print("PyTorch foi compilado com suporte a CUDA? " + ("Sim" if torch.cuda.is_available() else "Nao"))

if torch.cuda.is_available():
    # 2. Mostra a quantidade de GPUs disponiveis
    gpu_count = torch.cuda.device_count()
    print("Numero de GPUs disponiveis: " + str(gpu_count))

    # 3. Mostra o nome da GPU atual (geralmente a GPU 0)
    current_gpu_name = torch.cuda.get_device_name(0)
    print("Nome da GPU: " + current_gpu_name)

    # 4. Cria um tensor e o move para a GPU
    print("\n--- Testando operacao na GPU ---")
    try:
        # Cria um tensor na CPU
        tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
        print("Tensor na CPU: " + str(tensor_cpu.device))

        # Move o tensor para a GPU
        tensor_gpu = tensor_cpu.to("cuda")
        print("Tensor movido para a GPU: " + str(tensor_gpu.device))

        # Realiza uma operacao simples na GPU
        resultado = tensor_gpu * 2
        print("Resultado da operacao na GPU: " + str(resultado))
        print("Teste de GPU concluido com sucesso!")

    except Exception as e:
        print("Ocorreu um erro durante o teste na GPU: " + str(e))
else:
    print("\nPyTorch nao consegue acessar a GPU. Verifique a instalacao do PyTorch e os drivers da NVIDIA.")
