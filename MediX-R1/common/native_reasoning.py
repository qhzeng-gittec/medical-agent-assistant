import re


SYSTEM_PROMPT = (
    "You are a medical multimodal assistant. Reason from the supplied image, research context, "
    "or clinical case. Do not invent observations. Give the final answer in natural language."
)
CASE_IMAGE_REFERENCE = re.compile(
    r"\b(?:is|are|was|were|been|as|also)\s+(?:shown|depicted|illustrated)\b|"
    r"\bshown\s+(?:below|in|here)\b|\b(?:figure|fig\.)\s*(?:below|[A-D]\b|\d)|"
    r"\b(?:see|the|following)\s+(?:illustration|photograph|diagram)\b|"
    r"\b(?:histopath\w*|radiological finding|x\s*-?\s*ray|MRI|CT|karyogram|smear)"
    r"[^.?!]{0,70}\bgiven below\b|\bgiven below is the histopathology\b|"
    r"\bthis\b[^.?!]{0,70}\bscan\b|\bdepicted on the plate\b|"
    r"<img\b|!\[[^\]]*\]\(", re.IGNORECASE
)


def missing_case_image_reason(row: dict) -> str | None:
    options = list(row["options"].values())
    if any(re.search(r"image_question|\b(?:figure|image)\s*[A-D1-4]\b", x, re.I) for x in options):
        return "missing_option_image"
    if CASE_IMAGE_REFERENCE.search(row["case_question"]):
        return "missing_referenced_image"
    return None


def source_explanation(text: str) -> str:
    # Citation placement varies; preserve the full source rather than cut clinical content.
    text = re.sub(r"^\s*Answer\s*[:.-]?\s*[A-D]\s*[.)]?\s*", "", text)
    return " ".join(text.split()).strip()


def make_example(row: dict, task: str, image_path: str | None = None) -> dict:
    if task == "vqa":
        user_text = (
            f"Medical image question: {row['question']}\n\n"
            "Reason from the visible findings and answer the question in natural language."
        )
        reasoning = row.get("rationale", "")
        reference = row["answer"]
        final = f"The answer is {reference.rstrip('.')}."
        group = row["image_id"]
        provenance = "teacher_visible_evidence_not_expert_verified"
    elif task == "context":
        user_text = (
            f"Medical research context:\n{row['context']}\n\nQuestion: {row['question']}\n\n"
            "Explain what the supplied study supports, preserving uncertainty and its scope. "
            "Answer the research question in natural language."
        )
        reasoning = row["explanation"].strip()
        reference = row["answer"]
        final = f"{reference.capitalize()}. {reasoning}"
        group = row["pubmed_id"]
        provenance = "original_abstract_conclusion_not_new_cot"
    elif task == "case":
        user_text = (
            f"Medical question:\n{row['open_question']}\n\n"
            "Explain the clinical basis using the supplied information. "
            "Give the supported conclusion in natural language."
        )
        reasoning = source_explanation(row["evidence"])
        reference = row["answer"].strip()
        final = reference
        group = row["source_id"]
        provenance = "original_dataset_explanation_not_expert_revalidated"
    else:
        raise ValueError(f"Unknown task: {task}")
    return {
        "source_id": row["id"], "source_group": group, "task": task,
        "image_path": image_path, "user_text": user_text,
        "reasoning_content": reasoning, "target": final, "reference": reference,
        "reasoning_source": provenance,
    }


def parse_completion(text: str) -> dict:
    # Generation starts after the template's existing <think> token.
    reasoning, separator, final = text.partition("</think>")
    reasoning = reasoning.removeprefix("<think>").strip()
    final = final.strip() if separator else ""
    return {
        "reasoning": reasoning, "final_answer": final,
        "response_complete": bool(separator and reasoning and final),
    }
