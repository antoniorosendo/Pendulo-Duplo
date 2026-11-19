import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def carregar_ultimo_csv(diretorio="../../CSV/pendulo/"):
    arquivos = glob.glob(os.path.join(diretorio, "pendulo_log_*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {diretorio}")
    return max(arquivos, key=os.path.getctime)

def processar_angulos(caminho_csv):
    print(f"Lendo dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv, sep=';')

    # Filtra frames onde algum dos pontos não foi detectado (valores zerados)
    df_valido = df[(df['x1'] != 0) & (df['x2'] != 0) & (df['x3'] != 0)].copy()

    if df_valido.empty:
        print("Aviso: Não há frames suficientes com os 3 pontos detectados simultaneamente.")
        return None

    # Cálculo de Theta 1 (Haste superior em relação à vertical -Y a partir da origem 0,0)
    # Pode usar tanto o ponto vermelho (x1, y1) quanto o verde (x3, y3) pois ambos estão na haste 1
    df_valido['theta1_rad'] = np.arctan2(df_valido['x1'], -df_valido['y1'])
    df_valido['theta1_deg'] = np.degrees(df_valido['theta1_rad'])

    # Cálculo de Theta 2 (Haste inferior em relação à vertical -Y a partir da articulação verde x3, y3)
    dx2 = df_valido['x2'] - df_valido['x3']
    dy2 = df_valido['y2'] - df_valido['y3']
    df_valido['theta2_rad'] = np.arctan2(dx2, -dy2)
    df_valido['theta2_deg'] = np.degrees(df_valido['theta2_rad'])

    # Salva um novo CSV processado com os ângulos prontos para o relatório
    caminho_saida = caminho_csv.replace(".csv", "_angulos.csv")
    df_valido.to_csv(caminho_saida, sep=';', index=False)
    print(f"Arquivo com ângulos salvo em: {caminho_saida}")

    return df_valido

def plotar_angulos(df):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Gráfico Theta 1
    ax1.plot(df['time'], df['theta1_deg'], color='#ff4d4d', linewidth=1.5, label=r'$\theta_1$ (Haste Superior)')
    ax1.set_ylabel('Ângulo (Graus)')
    ax1.set_title('EVOLUÇÃO TEMPORAL DOS ÂNGULOS DO PÊNDULO DUPLO')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Gráfico Theta 2
    ax2.plot(df['time'], df['theta2_deg'], color='#4da6ff', linewidth=1.5, label=r'$\theta_2$ (Haste Inferior)')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Ângulo (Graus)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        ultimo_csv = carregar_ultimo_csv()
        dados_angulos = processar_angulos(ultimo_csv)
        if dados_angulos is not None:
            plotar_angulos(dados_angulos)
    except Exception as e:
        print(f"Erro na análise: {e}")