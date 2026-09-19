"""The single place where a probability vector becomes a response dict (eval and serve both call it)."""


def view(task_type, labels, probs, values=None):
    probs = [float(p) for p in probs]
    conf = max(probs)
    if task_type == "choice":
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"choice": labels[best], "probabilities": dict(zip(labels, probs)), "confidence": conf}
    if task_type == "score":
        values = values or [float(l) for l in labels]
        return {"score": sum(p * v for p, v in zip(probs, values)),
                "probabilities": dict(zip(labels, probs)), "confidence": conf}
    if task_type == "noul":  # labels are ["true", "false"]
        p = probs[labels.index("true")]
        return {"probability": p, "confidence": max(p, 1 - p)}
    raise ValueError(f"unknown task type {task_type!r}")
