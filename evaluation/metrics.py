"""
metrics.py

Purpose:
    Computes evaluation metrics for critical neighborhood prediction in HabitaBot.

Dependencies:
    - sacrebleu
    - rouge_score
"""

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

def barrios_criticos_gt(df, año, top_n=10):
    """
    Returns the list of ground-truth critical neighborhoods for a given year,
    defined as those with deficit_habitacional > 2500, ordered by deficit.
    """
    fragment = df[(df["año"] == año) & (df["deficit_habitacional"] > 2500)]
    fragment = fragment.sort_values("deficit_habitacional", ascending=False).head(top_n)
    return fragment["barrio"].tolist()

def evaluar_bloomz_criticos(respuesta_limpia, barrios_gt):
    """
    Evaluates model predictions with several metrics.
    """
    set_pred = set([b.lower() for b in respuesta_limpia])
    set_gt = set([b.lower() for b in barrios_gt])
    aciertos = set_pred & set_gt
    recall = len(aciertos) / len(set_gt) if set_gt else 0
    precision = len(aciertos) / len(set_pred) if set_pred else 0
    f1 = 2 * recall * precision / (recall + precision + 1e-9) if (recall + precision) else 0
    accuracy = int([b.lower() for b in respuesta_limpia][:len(barrios_gt)] == [b.lower() for b in barrios_gt])
    bleu = BLEU().corpus_score([" ".join(respuesta_limpia)], [[" ".join(barrios_gt)]]).score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge = scorer.score(" ".join(barrios_gt), " ".join(respuesta_limpia))['rougeL'].fmeasure
    ranks = [i+1 for i, b in enumerate(respuesta_limpia) if b.lower() in set_gt]
    mrr = 1.0 / ranks[0] if ranks else 0.0
    print(f"Real:      {barrios_gt}")
    print(f"Predicho:  {respuesta_limpia}")
    print(f"Aciertos:  {aciertos}")
    print(f"Recall:    {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1:        {f1:.2%}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"BLEU:      {bleu:.2f}")
    print(f"ROUGE-L:   {rouge:.2%}")
    print(f"MRR:       {mrr:.2f}")
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "bleu": bleu,
        "rougeL": rouge,
        "mrr": mrr
    }

import re
import time

def construir_contexto_barrios_criticos(df, año, top_n=10):
    """
    Builds a context string with the deficit values for all neighborhoods in a year.
    """
    fragment = df[(df["año"] == año) & (df["deficit_habitacional"] > 2500)]
    fragment = fragment.sort_values("deficit_habitacional", ascending=False).head(top_n)
    contexto = "\n".join([f"{r.barrio}: {int(r.deficit_habitacional)}" for _, r in fragment.iterrows()])
    return contexto

def limpiar_lista_barrios_bloomz(respuesta):
    """
    Extracts the list of predicted neighborhoods from the model output.
    """
    splitters = [
        "sin repetir y sin agregar ningún texto extra.",
        "separados por coma, sin repetir y sin agregar ningún texto extra.",
        "Responde SOLO con la lista de barrios críticos, separados por coma.",
        "Responde SOLO con la lista de barrios críticos, separados por coma"
    ]
    for splitter in splitters:
        if splitter in respuesta:
            respuesta = respuesta.split(splitter)[-1].strip()
            break
    barrios = re.findall(r'Barrio_\d+', respuesta)
    return list(dict.fromkeys(barrios))  # Remove duplicates, keep order

def consultar_barrios_criticos(pipe, df, año, top_n=10, verbose=True):
    """
    Consults the fine-tuned model for the list of critical neighborhoods for a given year.
    """
    contexto = construir_contexto_barrios_criticos(df, año, top_n)
    prompt = (
        f"Pregunta: ¿Cuáles son los barrios con déficit habitacional crítico en {año}?\n"
        f"Contexto:\n{contexto}\n"
        "Categoría: Crítico >2500\n"
        "Responde SOLO con los nombres exactos de los barrios críticos del contexto, separados por coma, sin repetir y sin agregar ningún texto extra."
    )
    t0 = time.time()
    salida = pipe(prompt, max_new_tokens=60)[0]["generated_text"]
    t1 = time.time()
    if verbose:
        print("Prompt usado:\n", prompt)
        print("\nRespuesta cruda del modelo:\n", salida)
        print(f"\nTiempo de inferencia: {t1-t0:.2f} segundos")
    barrios_limpios = limpiar_lista_barrios_bloomz(salida)
    if verbose:
        print("\nBarrios críticos extraídos:", barrios_limpios)
    return barrios_limpios