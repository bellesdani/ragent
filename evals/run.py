import yaml
import httpx
import asyncio
import argparse

from pathlib import Path
from pydantic import BaseModel, Field


class EvalStep(BaseModel):
    question: str = Field(min_length=1)
    expected: list[str] = Field(min_length=1)


class EvalCase(BaseModel):
    agent: str = Field(min_length=1)
    conversation: list[EvalStep] = Field(min_length=1)


class EvalDataset(BaseModel):
    cases: list[EvalCase] = Field(min_length=1)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carga y ejecuta un dataset de evaluaciones")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="evals/datasets/quipi_eval.yaml",
        help="Ruta al dataset YAML",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="URL base del servicio RAGent",
    )
    return parser


def load_dataset(dataset_path: str | Path) -> EvalDataset:
    path = Path(dataset_path)
    loaded_content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(loaded_content, dict):
        raise ValueError("El dataset debe ser un objeto YAML con una clave 'cases'")

    return EvalDataset.model_validate(loaded_content)


def normalize_text(value: str) -> str:
    return value.casefold()


def build_step_result(step_index: int, question: str, response: str, expected: list[str]) -> dict:
    normalized_response = normalize_text(response)
    missing_expected = [
        expected_fragment
        for expected_fragment in expected
        if normalize_text(expected_fragment) not in normalized_response
    ]
    return {
        "step_index": step_index,
        "question": question,
        "response": response,
        "missing_expected": missing_expected,
        "error": None,
        "passed": not missing_expected,
    }


async def request_response(client: httpx.AsyncClient, agent: str, messages: list[dict[str, str]]) -> str:
    response = await client.post(
        "v1/chat/completions",
        json={
            "model": agent,
            "messages": messages,
            "stream": False,
        },
    )
    response.raise_for_status()

    response_content = response.json()
    choices = response_content.get("choices")
    return choices[0]["message"]["content"]


async def execute_dataset(dataset: EvalDataset, base_url: str) -> list[dict]:
    case_results = []

    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=60.0) as client:
        for case_index, case in enumerate(dataset.cases, start=1):
            messages = []
            step_results = []

            for step_index, step in enumerate(case.conversation, start=1):
                messages.append(
                    {
                        "role": "user",
                        "content": step.question,
                    }
                )

                try:
                    response = await request_response(client=client, agent=case.agent, messages=messages)
                    step_result = build_step_result(
                        step_index=step_index,
                        question=step.question,
                        response=response,
                        expected=step.expected,
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    step_result = {
                        "step_index": step_index,
                        "question": step.question,
                        "response": "",
                        "missing_expected": [],
                        "error": str(exc),
                        "passed": False,
                    }
                    step_results.append(step_result)
                    break

                step_results.append(step_result)

            case_results.append(
                {
                    "case_index": case_index,
                    "agent": case.agent,
                    "passed": all(step_result["passed"] for step_result in step_results),
                    "steps": step_results,
                }
            )

    return case_results


def print_summary(base_url: str, dataset_path: Path, case_results: list[dict]) -> None:
    passed_count = sum(1 for case_result in case_results if case_result["passed"])
    failed_count = len(case_results) - passed_count

    print("")
    print(f"Dataset: {dataset_path}")
    print(f"Base URL: {base_url}")
    print(f"Cases: {len(case_results)}")
    print("")

    for case_result in case_results:
        status = "PASS" if case_result["passed"] else "FAIL"
        print(f"[{status}] case={case_result['case_index']}")
        for step_result in case_result["steps"]:
            step_status = "PASS" if step_result["passed"] else "FAIL"
            print(f"[{step_status}]   step={step_result['step_index']} | question={step_result['question']}")
            if step_result["missing_expected"]:
                print(f"\t\t| missing_expected={step_result['missing_expected']}")
            if step_result["error"]:
                print(f"\t\t| error={step_result['error']}")
            if step_status == "FAIL" and step_result["response"]:
                print(f"\t\t| response={step_result['response']}")

    print("")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")


async def async_main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    case_results = await execute_dataset(dataset=dataset, base_url=args.base_url)

    print_summary(base_url=args.base_url, dataset_path=dataset_path, case_results=case_results)

    if any(not case_result["passed"] for case_result in case_results):
        return 1

    return 0



if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
