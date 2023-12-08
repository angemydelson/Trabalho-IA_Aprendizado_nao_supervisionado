import cv2
import numpy as np
import os
import tempfile

def carregar_imagens(caminho_pasta):
    imagens = []
    for nome_arquivo in os.listdir(caminho_pasta):
        if nome_arquivo.endswith(".png"):
            caminho_img = os.path.join(caminho_pasta, nome_arquivo)
            img = cv2.imread(caminho_img)
            imagens.append(img)
    return imagens

def aplicar_kmeans(imagem, k):
    # Converte a imagem para o formato adequado
    dados_imagem = imagem.reshape((-1, 3))

    # Converte para float32
    dados_imagem = np.float32(dados_imagem)

    # Define os critérios de parada (número máximo de iterações ou precisão)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Aplica o algoritmo k-médias
    _, rotulos, centros = cv2.kmeans(dados_imagem, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converte os centros de volta para uint8
    centros = np.uint8(centros)

    # Mapeia os pixels para os valores dos centros
    imagem_segmentada = centros[rotulos.flatten()]

    # Remodela para as dimensões originais da imagem
    imagem_segmentada = imagem_segmentada.reshape(imagem.shape)

    return imagem_segmentada

def calcular_propriedades_imagem(imagem):
    resolucao = imagem.shape[0] * imagem.shape[1]

    # Salvar temporariamente a imagem para calcular o tamanho do arquivo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
        cv2.imwrite(temp_img.name, imagem)
        tamanho_kb = os.path.getsize(temp_img.name) / 1024

    cores_unicas = len(np.unique(imagem.reshape(-1, imagem.shape[2]), axis=0))

    return resolucao, tamanho_kb, cores_unicas

def imprimir_informacoes_imagem(tipo, k, propriedades, resolucao):
    largura, altura = resolucao[1], resolucao[0]

    print(f"{tipo} K={k}:")
    print(f"  Resolução: {largura}x{altura} pixels")
    print(f"  Tamanho em KB: {propriedades[1]:.2f} KB")
    print(f"  Cores únicas: {propriedades[2]}\n")



def salvar_imagem(imagem, caminho_saida):
    cv2.imwrite(caminho_saida, imagem)

def main():
    caminho_pasta = "imagens_originais"
    valores_k = [1, 2, 5, 6, 8, 10, 15]  # Defina os valores de k conforme necessário

    for k in valores_k:
        for imagem in carregar_imagens(caminho_pasta):
            imagem_segmentada = aplicar_kmeans(imagem, k)

            # Salve a imagem resultante
            caminho_saida = f"imagens_geradas/image_k{k}.png"
            salvar_imagem(imagem_segmentada, caminho_saida)

            # Calcule e imprima as informações sobre as imagens
            propriedades_originais = calcular_propriedades_imagem(imagem)
            resolucao_originais = imagem.shape
            propriedades_segmentadas = calcular_propriedades_imagem(imagem_segmentada)

            imprimir_informacoes_imagem("Original", k, propriedades_originais, resolucao_originais)
            imprimir_informacoes_imagem("Segmentada", k, propriedades_segmentadas, resolucao_originais)

if __name__ == "__main__":
    main()