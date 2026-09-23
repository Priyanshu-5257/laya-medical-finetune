"""Optional Weights & Biases helpers (no-op if wandb missing or disabled)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def resolve_wandb_api_key() -> Optional[str]:
    """Prefer env; on Kaggle also try common User Secrets names (never print the value)."""
    for env_name in ("WANDB_API_KEY", "wandb_api_key"):
        v = os.environ.get(env_name)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in (
            "wandb_api_key",
            "WANDB_API_KEY",
            "wandb",
            "wandb-api-key",
            "wandb_key",
            "WANDB",
            "wb_api_key",
            "WANDB_KEY",
        ):
            try:
                v = client.get_secret(name)
            except Exception:
                continue
            if v:
                os.environ["WANDB_API_KEY"] = v
                print(f"wandb: loaded Kaggle secret '{name}'", flush=True)
                return v
    except Exception as e:
        print(f"wandb: Kaggle secrets unavailable ({e})", flush=True)
    return None


def maybe_init_wandb(conf: Dict[str, Any], run_config: Dict[str, Any], enabled: bool = True):
    """Init wandb on rank-0 only. Returns run or None."""
    wconf = conf.get("wandb") or {}
    if not enabled or not wconf.get("enabled", True):
        print("wandb: disabled", flush=True)
        return None
    key = resolve_wandb_api_key()
    if not key:
        print("wandb: no API key found (set WANDB_API_KEY or Kaggle secret wandb_api_key)", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("wandb: package not installed", flush=True)
        return None
    mode = wconf.get("mode") or os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        project=str(wconf.get("project", "laya-medical")),
        name=wconf.get("name") or wconf.get("run_name"),
        entity=wconf.get("entity") or None,
        config=run_config,
        mode=mode,
        reinit=True,
    )
    print(f"wandb: started run {run.name} ({run.url})", flush=True)
    return run


def wandb_log(run, data: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    run.log(data, step=step)


def wandb_finish(run) -> None:
    if run is None:
        return
    run.finish()
