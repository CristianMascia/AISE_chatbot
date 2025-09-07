import os
import json
import time
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Opik
from opik import configure, track, opik_context, Opik

import backend

#setup
configure(
    use_local=False,
    api_key=os.getenv("OPIK_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)

NOTEBOOK_NAME = "cloud"
GROUNDTRUTH_PATH = Path("groundtruth_cache.json")


EVAL_LLM_MODEL = "gemini-2.0-flash"
EVAL_EMB_MODEL = "models/text-embedding-004"

eval_llm = ChatGoogleGenerativeAI(
    model=EVAL_LLM_MODEL,
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
eval_emb = GoogleGenerativeAIEmbeddings(
    model=EVAL_EMB_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

REQUEST_DELAY = 10  # secondi
OUTPUT_XLSX = "risultati_ragasopik.xlsx"


@track
def run_rag_for_question(rag_chain, question: str):
    """Esegue la catena RAG e restituisce answer + contexts per la singola domanda."""
    response = rag_chain.invoke({"question": question})
    ans = (response.get("answer") or "").strip()
    source_docs = response.get("source_documents", []) or []
    contexts = [doc.page_content for doc in source_docs]
    return {"answer": ans, "contexts": contexts}


@track
def run_ragas(dataset: Dataset, eval_llm, eval_emb):
    """Esegue RAGAS sulle quattro metriche e restituisce risultati + riepilogo medie."""
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_emb,
    )
    df = result.to_pandas()
    summary = {
        "context_precision_mean": float(df["context_precision"].mean()),
        "context_recall_mean": float(df["context_recall"].mean()),
        "faithfulness_mean": float(df["faithfulness"].mean()),
        "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
        "num_rows": int(len(df)),
    }
    return {"result": result, "summary": summary}


@track 
def main(
    notebook_name: str = NOTEBOOK_NAME,
    groundtruth_path: str = str(GROUNDTRUTH_PATH),
    request_delay: int = REQUEST_DELAY,
    eval_llm_model: str = EVAL_LLM_MODEL,
    eval_emb_model: str = EVAL_EMB_MODEL,
    output_xlsx: str = OUTPUT_XLSX,
):
    
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    questions = [x["question"] for x in gold]
    ground_truth = [x["ground_truth"] for x in gold]

    
    rag = backend.prepare_rag_chain(notebook_name)

    
    answers, contexts = [], []
    for idx, q in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Domanda: {q}")
        out = run_rag_for_question(rag, q)
        answers.append(out["answer"])
        contexts.append(out["contexts"])
        print(f"  ↳ Risposta: {out['answer'][:80]}...")
        time.sleep(request_delay)

    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,
    })


    ragas_out = run_ragas(dataset, eval_llm, eval_emb)
    result = ragas_out["result"]
    summary = ragas_out["summary"]

    
    df = result.to_pandas()
    df.to_excel(output_xlsx, index=False)
    print(f"✅ Risultati salvati in {output_xlsx}")

   
    opik_context.update_current_trace(
        tags=["evaluation", "ragas", notebook_name],
        feedback_scores=[
            {"name": "ragas_context_precision_mean", "value": summary["context_precision_mean"]},
            {"name": "ragas_context_recall_mean",    "value": summary["context_recall_mean"]},
            {"name": "ragas_faithfulness_mean",      "value": summary["faithfulness_mean"]},
            {"name": "ragas_answer_relevancy_mean",  "value": summary["answer_relevancy_mean"]},
            {"name": "ragas_num_questions",          "value": float(len(questions))},
        ],
    )

    
    return {
        "output_file": output_xlsx,
        "metrics_summary": summary,
        "num_questions": len(questions),
        "models": {"eval_llm": eval_llm_model, "eval_embeddings": eval_emb_model},
        "notebook": notebook_name,
    }


if __name__ == "__main__":
    try:
        main()
    finally:
        #garantisce che tutti i log vengano inviati ad Opik prima dell'uscita
        Opik().flush()
