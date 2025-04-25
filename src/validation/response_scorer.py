"""
Response Scorer

Calculates BLEU, ROUGE-L, and F1 score between chatbot output and retrieved references.
"""

from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import re

def clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()

def tokenize(text: str) -> List[str]:
    return clean_text(text).split()

def compute_bleu(candidate: str, references: List[str]) -> float:
    try:
        ref_tokens = [tokenize(ref) for ref in references]
        cand_tokens = tokenize(candidate)
        smoothie = SmoothingFunction().method4
        return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)
    except Exception:
        return 0.0

def compute_rouge(candidate: str, references: List[str]) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, candidate)['rougeL'].fmeasure for ref in references]
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0

def compute_f1(candidate: str, references: List[str]) -> float:
    try:
        cand_tokens = tokenize(candidate)
        ref_tokens = tokenize(" ".join(references))
        common = set(cand_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(cand_tokens) if cand_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    except Exception:
        return 0.0