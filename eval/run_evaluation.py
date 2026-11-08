import argparse
import csv
import json
import time
from pathlib import Path

from src.retrieve import Retriever
from src.generate import generate_answer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

CSV_FILE = RESULTS_DIR / "evaluation_results.csv"
SUMMARY_FILE = RESULTS_DIR / "automated_summary.json"

FIELDNAMES = [
    "id",
    "category",
    "question",
    "answerable",
    "gold_chunk_ids",
    "retrieved_chunk_ids",
    "retrieved_pages",
    "hit_at_1",
    "hit_at_4",
    "answer",
    "answer_score_0_2",
    "citation_correct_0_1",
    "status",
]


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_existing_results():
    if not CSV_FILE.exists():
        return {}

    results = {}

    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("status") == "completed":
                results[row["id"]] = row

    return results


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in results.values():
            writer.writerow(row)


def is_quota_error(error):
    message = str(error).lower()

    indicators = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "503",
        "unavailable",
        "500",
        "internal server error",
    ]

    return any(term in message for term in indicators)


def generate_with_retry(question, passages, max_retries=4):
    for attempt in range(max_retries + 1):
        try:
            answer = generate_answer(question, passages)

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The generation model returned an empty answer.")

            if "429 RESOURCE_EXHAUSTED" in answer:
                raise RuntimeError(answer)

            return answer

        except Exception as error:
            if attempt == max_retries:
                raise

            if not is_quota_error(error):
                raise

            wait_time = min(15 * (2 ** attempt), 120)

            print(
                f"Temporary API error. "
                f"Retrying in {wait_time} seconds "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)


def evaluate_question(item, retriever):
    question = item["question"]

    retrieved = retriever.search(question, top_k=6)

    retrieved_ids = [str(p["chunk_id"]) for p in retrieved]
    retrieved_pages = [p["page"] for p in retrieved]

    answer = generate_with_retry(question, retrieved)

    answerable = item["answerable"]
    gold_ids = [str(x) for x in item["gold_chunk_ids"]]

    if answerable:
        hit_at_1 = int(bool(set(retrieved_ids[:1]) & set(gold_ids)))
        hit_at_4 = int(bool(set(retrieved_ids[:4]) & set(gold_ids)))
    else:
        hit_at_1 = ""
        hit_at_4 = ""

    refusal_phrase = "I could not find this in the loaded documents."

    if answerable:
        status = "completed"
    elif refusal_phrase in answer:
        status = "completed"
    else:
        status = "completed"

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "answerable": answerable,
        "gold_chunk_ids": json.dumps(gold_ids),
        "retrieved_chunk_ids": json.dumps(retrieved_ids),
        "retrieved_pages": json.dumps(retrieved_pages),
        "hit_at_1": hit_at_1,
        "hit_at_4": hit_at_4,
        "answer": answer,
        "answer_score_0_2": "",
        "citation_correct_0_1": "",
        "status": status,
    }


def calculate_summary(results, questions):
    completed = [
        results[item["id"]]
        for item in questions
        if item["id"] in results
    ]

    answerable_results = [
    row for row in completed
    if str(row["answerable"]).lower() == "true"
    ]

    unanswerable_results = [
    row for row in completed
    if str(row["answerable"]).lower() == "false"
    ]

    def mean(values):
        return sum(values) / len(values) if values else None

    hit1 = [
        int(row["hit_at_1"])
        for row in answerable_results
        if row["hit_at_1"] != ""
    ]

    hit4 = [
        int(row["hit_at_4"])
        for row in answerable_results
        if row["hit_at_4"] != ""
    ]

    refusal_phrase = "I could not find this in the loaded documents."

    refusal_correct = [
        int(refusal_phrase in row["answer"])
        for row in unanswerable_results
    ]

    return {
        "total_questions": len(questions),
        "completed_questions": len(completed),
        "failed_or_pending_questions": len(questions) - len(completed),
        "answerable_questions_completed": len(answerable_results),
        "unanswerable_questions_completed": len(unanswerable_results),
        "hit_at_1": mean(hit1),
        "hit_at_4": mean(hit4),
        "refusal_accuracy": mean(refusal_correct),
        "answer_correctness": "Pending human scoring (0-2)",
        "citation_correctness": "Pending human scoring (0-1)",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Seconds to wait between Gemini requests.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of passages retrieved per question.",
    )

    args = parser.parse_args()

    questions = load_questions()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading retrieval resources...")
    retriever = Retriever()

    print(f"Loaded {len(retriever.metadata)} document chunks.")
    print("Dense retrieval: Ready")
    print("BM25 retrieval: Ready")

    results = load_existing_results()

    if results:
        print(f"\nResuming evaluation: {len(results)} questions already completed.")
    else:
        print("\nStarting a new evaluation.")

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]

        if question_id in results:
            print(f"\n[{index}/{len(questions)}] {question_id}: Already completed. Skipping.")
            continue

        print(f"\n[{index}/{len(questions)}] {question_id}: {item['question']}")

        try:
            row = evaluate_question(item, retriever)
            results[question_id] = row

            print(f"Retrieved pages: {row['retrieved_pages']}")
            print(f"Hit@1: {row['hit_at_1']} | Hit@4: {row['hit_at_4']}")
            print(f"Answer: {row['answer']}")

            # Save immediately after each successful question
            save_results(results)

        except Exception as error:
            print(f"FAILED: {error}")
            print("This question will remain pending and can be retried later.")

        if index < len(questions):
            time.sleep(args.delay)

    summary = calculate_summary(results, questions)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== AUTOMATED EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print(f"\nDetailed results: {CSV_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()