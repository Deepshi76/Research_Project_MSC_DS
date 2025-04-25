# src/validation/hallucination_checker.py

"""
Hallucination Checker

Flags low‐confidence retrieval based on a numeric threshold.
"""

def check_hallucination(halluc_score: float, threshold: float = 0.5) -> bool:
    """
    Args:
      halluc_score – the average FAISS similarity [0,1]
      threshold    – below this is considered a hallucination

    Returns:
      True if halluc_score < threshold (i.e. low confidence), False otherwise.
    """
    return halluc_score < threshold
