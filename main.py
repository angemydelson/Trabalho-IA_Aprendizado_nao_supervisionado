import cv2
import numpy as np
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt


def carregar_imagens(caminho_pasta):
    # Lista para armazenar as imagens carregadas
    imagens = []
    try:
        # Itera sobre os arquivos na pasta fornecida
        for nome_arquivo in os.listdir(caminho_pasta):
            # Verifica se o arquivo tem a extensão .png
            if nome_arquivo.endswith(".png"):
                # Constrói o caminho completo para a imagem
                caminho_img = os.path.join(caminho_pasta, nome_arquivo)                
                try:
                    # Tenta ler a imagem usando OpenCV
                    img = cv2.imread(caminho_img)                    
                    # Verifica se a leitura foi bem-sucedida
                    if img is not None:
                        # Adiciona a imagem à lista
                        imagens.append(img)
                    else:
                        # Imprime uma mensagem de erro se a leitura falhar
                        print(f"Erro ao ler a imagem: {caminho_img}")                        
                except Exception as e:
                    # Imprime uma mensagem de erro se ocorrer uma exceção ao abrir a imagem
                    print(f"Erro ao abrir a imagem {caminho_img}: {e}")
    except Exception as e:
        # Imprime uma mensagem de erro se ocorrer uma exceção ao listar os arquivos na pasta
        print(f"Erro ao listar arquivos em {caminho_pasta}: {e}")
    return imagens



def aplicar_kmeans(imagem, k):
    try:
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

    except Exception as e:
        print(f"Erro ao aplicar k-means: {e}")
        return None



def calcular_propriedades_imagem(imagem):
    try:
        resolucao = imagem.shape[0] * imagem.shape[1]

        # Salvar temporariamente a imagem para calcular o tamanho do arquivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            cv2.imwrite(temp_img.name, imagem)
            tamanho_kb = os.path.getsize(temp_img.name) / 1024
        cores_unicas = len(np.unique(imagem.reshape(-1, imagem.shape[2]), axis=0))
        return resolucao, tamanho_kb, cores_unicas
    
    except Exception as e:
        print(f"Erro ao calcular propriedades da imagem: {e}")
        return None, None, None



def imprimir_informacoes_imagem(tipo, k, propriedades, resolucao):
    try:
        # Obtém a largura e altura da imagem da resolução
        largura, altura = resolucao[1], resolucao[0]

        # Lista para armazenar as informações da imagem formatadas
        info_string = []

        # Adiciona informações formatadas à lista
        if k == 0:
            info_string.append(f"{tipo}")
            info_string.append(f"  Resolução: {largura}x{altura} pixels")
            info_string.append(f"  Tamanho em KB: {propriedades[1]:.2f} KB")
            info_string.append(f"  Cores únicas: {propriedades[2]}\n")
        else:
            info_string.append(f"{tipo} K={k}:")
            info_string.append(f"  Resolução: {largura}x{altura} pixels")
            info_string.append(f"  Tamanho em KB: {propriedades[1]:.2f} KB")
            info_string.append(f"  Cores únicas: {propriedades[2]}\n")

        # Retorna a lista com informações formatadas
        return info_string

    except Exception as e:
        # Retorna mensagem de erro se ocorrer uma exceção ao gerar informações da imagem
        return f"Erro ao gerar informações da imagem: {e}"



def plotar_graficos(titulo, valores_k, resolucoes, tamanhos_kb, cores_unicas):
    try:
        # Configuração da figura e dos subplots
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        
        # Gráfico de Resolução
        plt.plot(valores_k, resolucoes, marker='o')
        plt.title('Resolução')
        plt.xlabel('K')
        plt.ylabel('Resolução (pixels)')

        # Gráfico de Tamanho do Arquivo
        plt.subplot(2, 2, 2)
        plt.plot(valores_k, tamanhos_kb, marker='o')
        plt.title('Tamanho do Arquivo')
        plt.xlabel('K')
        plt.ylabel('Tamanho (KB)')

        # Gráfico de Cores Únicas
        plt.subplot(2, 2, 3)
        plt.plot(valores_k, cores_unicas, marker='o')
        plt.title('Cores Únicas')
        plt.xlabel('K')
        plt.ylabel('Número de Cores Únicas')

        # Título geral e ajuste do layout
        plt.suptitle(titulo)
        plt.tight_layout()

        # Exibe os gráficos
        plt.show()

    except Exception as e:
        # Imprime mensagem de erro se ocorrer uma exceção ao plotar os gráficos
        print(f"Erro ao plotar gráficos: {e}")

        

def salvar_imagem(imagem, caminho_saida):
    try:
        # Tenta salvar a imagem usando o OpenCV
        cv2.imwrite(caminho_saida, imagem)
        
        # Mensagem de sucesso se a operação for bem-sucedida
        print(f"Imagem salva com sucesso em {caminho_saida}")

    except Exception as e:
        print(f"Erro ao salvar a imagem em {caminho_saida}: {e}")

    
    
def criar_pdf(informacoes, pdf_path="imagens_geradas/Resultados_Imagem6/informacoes_imagens.pdf"):
    try:
        pdf = canvas.Canvas(pdf_path, pagesize=letter)
        # Definir o estilo da fonte
        pdf.setFont("Helvetica", 12)
        # Título
        pdf.drawCentredString(letter[0] / 2, pdf._pagesize[1] - 50, "Informações das Imagens")
        # Posição inicial Y
        y_position = pdf._pagesize[1] - 70
        # Espaço entre as informações
        line_height = 12

        for info in informacoes:
            # Espaço entre blocos de informação
            y_position -= line_height

            for line in info:
                pdf.drawString(100, y_position, line)
                y_position -= line_height

            # Adicionar espaço entre conjuntos de informações
            y_position -= line_height * 2

        # Salvar o PDF
        pdf.save()
        print(f"PDF criado com sucesso: {pdf_path}")

    except Exception as e:
        print(f"Erro ao criar o PDF: {e}")


    
def main():
    try:
        caminho_pasta = "imagens_originais"
        valores_k = [2, 5, 10, 15, 150, 200, 300]
        resolucoes, tamanhos_kb, cores_unicas = [], [], []
        verificador = 1
        informacoes = []
        
        for k in valores_k:
            for imagem in carregar_imagens(caminho_pasta):
                imagem_segmentada = aplicar_kmeans(imagem, k)
                
                # Salve a imagem resultante
                caminho_saida = f"imagens_geradas/Resultados_Imagem6/image_k{k}.png"
                salvar_imagem(imagem_segmentada, caminho_saida)
                
                # Calcule e imprima as informações sobre as imagens
                propriedades_originais = calcular_propriedades_imagem(imagem)
                resolucao_originais = imagem.shape
                propriedades_segmentadas = calcular_propriedades_imagem(imagem_segmentada)
                
                if verificador == 1:
                    informacoes.append(imprimir_informacoes_imagem("Informações da imagem original: ", 0, propriedades_originais, resolucao_originais))
                    verificador += 1
                    
                imprimir_informacoes_imagem("Informações da imagem resultante: ", k, propriedades_segmentadas, resolucao_originais)
                informacoes.append(imprimir_informacoes_imagem("Informações da imagem resultante: ", k, propriedades_segmentadas, resolucao_originais))
                resolucoes.append(resolucao_originais[0] * resolucao_originais[1])
                tamanhos_kb.append(propriedades_segmentadas[1])
                cores_unicas.append(propriedades_segmentadas[2])

        plotar_graficos('Análise K-means', valores_k, resolucoes, tamanhos_kb, cores_unicas)
        criar_pdf(informacoes)

    except Exception as e:
        print(f"Erro na execução da função main: {e}")



if __name__ == "__main__":
    main()