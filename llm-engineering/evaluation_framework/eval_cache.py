# eval_cache.py
import os
import json
import hashlib
from .models import EvalScore

CACHE_FILE = ".eval_cache.json"

class EvalCache:
    def __init__(self):
        self.cache = {}
        self.load()

    def load(self):
        # Clear cache if clear environment variable is set
        if os.environ.get("CLEAR_EVAL_CACHE") == "true":
            if os.path.exists(CACHE_FILE):
                try:
                    os.remove(CACHE_FILE)
                except Exception:
                    pass
            self.cache = {}
            return

        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def save(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def _get_key(self, input_text, model_output, reference_output, criterion, rater_idx):
        ref = reference_output if reference_output else ""
        content = f"{input_text}|||{model_output}|||{ref}|||{criterion}|||{rater_idx}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, input_text, model_output, reference_output, criterion, rater_idx):
        key = self._get_key(input_text, model_output, reference_output, criterion, rater_idx)
        if key in self.cache:
            data = self.cache[key]
            return EvalScore(
                criterion=data["criterion"],
                score=data["score"],
                reasoning=data["reasoning"]
            )
        return None

    def set(self, input_text, model_output, reference_output, criterion, rater_idx, eval_score):
        key = self._get_key(input_text, model_output, reference_output, criterion, rater_idx)
        self.cache[key] = {
            "criterion": eval_score.criterion,
            "score": eval_score.score,
            "reasoning": eval_score.reasoning
        }
        self.save()

    def _get_pairwise_key(self, input_text, output_a, output_b, reference_output):
        ref = reference_output if reference_output else ""
        content = f"pairwise|||{input_text}|||{output_a}|||{output_b}|||{ref}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_pairwise(self, input_text, output_a, output_b, reference_output):
        key = self._get_pairwise_key(input_text, output_a, output_b, reference_output)
        return self.cache.get(key)

    def set_pairwise(self, input_text, output_a, output_b, reference_output, result):
        key = self._get_pairwise_key(input_text, output_a, output_b, reference_output)
        self.cache[key] = result
        self.save()

# Global cache instance
eval_cache = EvalCache()
