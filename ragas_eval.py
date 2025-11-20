import os, json, time
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import backend

NOTEBOOK_NAME = "cloud"  #example notebook name
GROUNDTRUTH_PATH = Path("groundtruth_cache.json") #json with the ground truth

# Gemini models for RAGAS evaluation
eval_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
eval_emb = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


REQUEST_DELAY = 10 

def main():
    with open(GROUNDTRUTH_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)
    questions = [x["question"] for x in gold]
    ground_truth = [x["ground_truth"] for x in gold]

    rag = backend.prepare_rag_chain(NOTEBOOK_NAME)

    answers, contexts = [], []
    for idx, q in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Question: {q}")
        
        response = rag.invoke({"question": q})
        ans = response.get("answer", "").strip()
        source_docs = response.get("source_documents", [])
        
        answers.append(ans)
        
        contexts.append([doc.page_content for doc in source_docs])
        
        print(f"  ↳ Answer: {ans[:80]}...")
        time.sleep(REQUEST_DELAY)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,
    })

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_emb,
    )

    df = result.to_pandas()
    output_file = "ragas_results_3.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ Results saved in {output_file}")

if __name__ == "__main__":
    main()
