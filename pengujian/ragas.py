import os, json, logging, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.embeddings import LangchainEmbeddingsWrapper

try:
    from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
except ImportError:
    # fallback kompatibel dengan env saat ini
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
HF_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MAX_TOKENS = int(os.getenv("RAGAS_LLM_MAX_TOKENS", "2048"))
MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
BATCH_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "20"))
MAX_WAIT = int(os.getenv("RAGAS_MAX_WAIT", "120"))
MAX_CONTEXTS = int(os.getenv("RAGAS_MAX_CONTEXTS", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "900"))
MAX_RESPONSE_SENTENCES = int(os.getenv("RAGAS_MAX_RESPONSE_SENTENCES", "4"))
MAX_RESPONSE_CHARS = int(os.getenv("RAGAS_MAX_RESPONSE_CHARS", "1000"))
MAX_QUESTION_CHARS = int(os.getenv("RAGAS_MAX_QUESTION_CHARS", "300"))
MAX_STATEMENTS = int(os.getenv("RAGAS_MAX_STATEMENTS", "6"))
USE_CONTEXT_METRICS = os.getenv("RAGAS_USE_CONTEXT_METRICS", "1").lower() in {"1", "true", "yes"}

if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY di .env dulu ya.")

def load_dataset(filename="dataEvaluasi.json") -> EvaluationDataset:
    path = Path(__file__).parent / filename
    raw = json.loads(path.read_text(encoding="utf-8"))

    samples = []
    for r in raw:
        user_input = r.get("user_input") or r.get("question") or ""
        retrieved_contexts = r.get("retrieved_contexts")
        if retrieved_contexts is None:
            retrieved_contexts = r.get("contexts", [])
        if isinstance(retrieved_contexts, str):
            retrieved_contexts = [retrieved_contexts]

        response = r.get("response") or r.get("answer") or ""

        ref = r.get("reference")
        if ref is None:
            ref = r.get("ground_truths", [])
        reference = "; ".join(map(str, ref)) if isinstance(ref, list) else str(ref or "")

        samples.append(
            SingleTurnSample(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                response=response,
                reference=reference,
            )
        )
    return EvaluationDataset(samples=samples)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _truncate_chars(text: str, max_chars: int) -> str:
    text = _normalize_ws(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

def _truncate_sentences(text: str, max_sentences: int) -> str:
    text = _normalize_ws(text)
    if max_sentences <= 0 or not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return text
    return " ".join(parts[:max_sentences])

def compress_sample_for_eval(sample: SingleTurnSample) -> SingleTurnSample:
    contexts = sample.retrieved_contexts or []
    contexts = [
        _truncate_chars(c, MAX_CONTEXT_CHARS)
        for c in contexts[:MAX_CONTEXTS]
        if _normalize_ws(c)
    ]

    response = _truncate_sentences(sample.response or "", MAX_RESPONSE_SENTENCES)
    response = _truncate_chars(response, MAX_RESPONSE_CHARS)
    user_input = _truncate_chars(sample.user_input or "", MAX_QUESTION_CHARS)

    return SingleTurnSample(
        user_input=user_input,
        retrieved_contexts=contexts,
        response=response,
        reference=sample.reference,
    )

def choose_metrics():
    answer_relevancy.strictness = 1
    faithfulness.statement_generator_prompt.instruction = (
        "Given a question and an answer, analyze the complexity of each sentence in the answer. "
        "Break down each sentence into one or more fully understandable statements. "
        f"Return at most {MAX_STATEMENTS} statements. "
        "Ensure that no pronouns are used in any statement. Format the outputs in JSON."
    )
    metrics = [faithfulness, answer_relevancy]
    if USE_CONTEXT_METRICS:
        metrics.extend([context_precision, context_recall])
    return metrics

def main():
    dataset = load_dataset("dataEvaluasi.json")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    model_name = MODEL.replace("models/", "")

    llm = llm_factory(
        model_name,
        provider="openai",
        client=client,
        max_tokens=LLM_MAX_TOKENS,
    )

    lc_embeddings = LCHuggingFaceEmbeddings(model_name=HF_MODEL)
    embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ragas.retry").setLevel(logging.DEBUG)
    logging.getLogger("TENACITYRetry").setLevel(logging.DEBUG)

    run_config = RunConfig(
        max_workers=MAX_WORKERS,
        max_retries=MAX_RETRIES,
        max_wait=MAX_WAIT,
        timeout=BATCH_TIMEOUT,
        log_tenacity=True,
    )

    metrics = choose_metrics()
    eval_samples = [compress_sample_for_eval(s) for s in dataset.samples]
    eval_dataset = EvaluationDataset(samples=eval_samples)
    print(f"\n=== Evaluating {len(eval_samples)} samples (single batch) ===")

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )
    print(result)

    frame = result.to_pandas()
    print(frame)

    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in frame.columns]
    if metric_cols:
        print("\n=== Rata-rata skor ===")
        print(frame[metric_cols].mean(numeric_only=True))

if __name__ == "__main__":
    main()
