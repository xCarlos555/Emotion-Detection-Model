import pandas as pd
import random


aus = sorted([1, 2, 4, 5, 6, 7, 9, 12, 15, 17, 20, 23, 26])
au_labels = [f"AU{i}" for i in aus]

hierarquia_emocoes = ["Happy", "Surprise", "Angry", "Sad", "Fear", "Disgust"]

emocoes_aus_com_condicoes = {
    "Happy": [{"positivos": [6, 12], "negativos": [20, 26], "opcionais": [4, 5, 17]}],

    "Surprise": [{"positivos": [5, 26], "negativos": [20], "opcionais": [1, 2]}],

    "Angry": [{"positivos": [4, 23, 7], "negativos": [26], "opcionais": [5, 6, 9, 17]}],

    "Sad": [{"positivos": [4, 15], "negativos": [7, 9, 12, 23, 26], "opcionais": []}],

    "Fear": [{"positivos": [2, 20, 26], "negativos": [9], "opcionais": [15, 17]}],

    "Disgust": [{"positivos": [6, 9, 17], "negativos": [23], "opcionais": [7, 15]}]
}

alvo_por_classe = 2000
dados = []

for emocao in hierarquia_emocoes:
    combinacoes = emocoes_aus_com_condicoes[emocao]

    for comb in combinacoes:
        positivos = comb["positivos"]
        negativos = comb["negativos"]
        opcionais = comb.get("opcionais", [])

        exemplos_gerados = 0
        while exemplos_gerados < alvo_por_classe:
            linha = []
            linha_dict = {}
            for au in aus:
                if au in positivos:
                    val = round(random.uniform(95.0, 100.0), 2)
                elif au in negativos:
                    val = round(random.uniform(0.0, 85.0), 2)
                elif au in opcionais:
                    val = round(random.uniform(0.0, 100.0), 2)
                else:
                    val = round(random.uniform(0.0, 85.0), 2)
                linha.append(val)
                linha_dict[au] = val

            satisfaz_emocao_anterior = False
            for emocao_ant in hierarquia_emocoes:
                if emocao_ant == emocao:
                    break
                for comb_ant in emocoes_aus_com_condicoes.get(emocao_ant, []):
                    pos_ant = comb_ant["positivos"]
                    neg_ant = comb_ant["negativos"]
                    if all(linha_dict[au] >= 95.0 for au in pos_ant) and all(linha_dict[au] <= 85.0 for au in neg_ant):
                        satisfaz_emocao_anterior = True
                        break
                if satisfaz_emocao_anterior:
                    break

            if satisfaz_emocao_anterior:
                continue

            satisfaz_positivos = all(linha_dict[au] >= 95.0 for au in positivos)
            satisfaz_negativos = all(linha_dict[au] <= 85.0 for au in negativos)
            if not (satisfaz_positivos and satisfaz_negativos):
                continue

            linha.append(emocao)
            dados.append(linha)
            exemplos_gerados += 1

df = pd.DataFrame(dados, columns=au_labels + ["Emotion"])
df.to_csv("dataset_AUs.csv", sep=';', index=False)
