from datetime import timedelta
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from t2i_eval.eval.geneval.evaluator import GenevalEvaluator
from t2i_eval.model.diffusers_model import DiffusersModel

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
    evaluator = GenevalEvaluator(
        # category="counting",
        # num_samples=10,
        accelerator=accelerator,
        sample_dir=str(BASE_DIR / "geneval_samples"),
    )
    model = DiffusersModel(
        pretrained="sd2-community/stable-diffusion-2-1",
        device="cuda",
        dtype="float16",
        disable_safety_checker=True,
    )

    eval_results: dict[str, Any] | None = evaluator.evaluate(model)
    if eval_results is not None:
        # save eval results to a json file
        import json

        with open(BASE_DIR / "geneval_eval_results.json", "w") as fp:
            json.dump(eval_results, fp, indent=4)
