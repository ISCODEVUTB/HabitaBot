"""
prompt_generation.py

Purpose:
    Generates prompt-completion training examples for fine-tuning based on the Cartagena housing deficit data.
    This script is the second step in the HabitaBot pipeline.

Main functions:
    - ejemplo_barrios_criticos: Example for top barrios with critical housing deficit per year.
    - ejemplo_deficit_barrio: Example for housing deficit by neighborhood in a year.
    - ejemplo_top_barrio: Example for the single neighborhood with the highest deficit.

Dependencies:
    - pandas
    - json
    - data_processing

Output:
    - Saves cartagena_rag_finetune.jsonl in the finetune/ directory.
"""

import pandas as pd
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_processing import load_data, clean_columns

def ejemplo_barrios_criticos(df, año):
    fragment = df[df["año"] == año][["barrio", "deficit_habitacional"]]
    fragment = fragment.sort_values("deficit_habitacional", ascending=False)
    criticos = fragment[fragment["deficit_habitacional"] > 2500]
    contexto = "\n".join([f"{r.barrio}: {int(r.deficit_habitacional)}" for _, r in fragment.iterrows()])
    respuesta = ", ".join(criticos.barrio.tolist())
    return {
        "prompt": f"Pregunta: ¿Cuáles son los barrios con déficit habitacional crítico en {año}?\n"
                  f"Contexto:\n{contexto}\n"
                  "Categoría: Crítico >2500\nResponde SOLO con la lista de barrios críticos, separados por coma.",
        "completion": respuesta if respuesta else "No hay barrios críticos en ese año."
    }

def ejemplo_deficit_barrio(df, barrio, año):
    fragment = df[(df["año"] == año) & (df["barrio"] == barrio)]
    if fragment.empty:
        return None
    deficit = int(fragment.iloc[0]["deficit_habitacional"])
    contexto = f"{barrio} en {año}: {deficit}"
    return {
        "prompt": f"Pregunta: ¿Cuál es el déficit habitacional de {barrio} en {año}?\n"
                  f"Contexto:\n{contexto}",
        "completion": f"{deficit}"
    }

def ejemplo_top_barrio(df, año):
    fragment = df[df["año"] == año][["barrio", "deficit_habitacional"]]
    fragment = fragment.sort_values("deficit_habitacional", ascending=False)
    top = fragment.iloc[0]
    contexto = "\n".join([f"{r.barrio}: {int(r.deficit_habitacional)}" for _, r in fragment.head(5).iterrows()])
    return {
        "prompt": f"Pregunta: ¿Cuál es el barrio con mayor déficit habitacional en {año}?\n"
                  f"Contexto:\n{contexto}\nResponde SOLO con el nombre del barrio.",
        "completion": top.barrio
    }

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datos_sinteticos_cartagena_3200.csv')
    data_path = os.path.abspath(data_path)
    df = load_data(data_path)
    df = clean_columns(df)
    ejemplos = []
    for año in sorted(df["año"].unique()):
        ejemplos.append(ejemplo_barrios_criticos(df, año))
        ejemplos.append(ejemplo_top_barrio(df, año))
    for idx, row in df.sample(600, random_state=42).iterrows():
        ej = ejemplo_deficit_barrio(df, row["barrio"], row["año"])
        if ej:
            ejemplos.append(ej)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'finetune', 'cartagena_rag_finetune.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for ej in ejemplos:
            f.write(json.dumps(ej, ensure_ascii=False) + '\n')
    print(f"Generados {len(ejemplos)} ejemplos RAG-style para fine-tuning.")