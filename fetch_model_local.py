"""
IMARA AI - Download model weights to run 100% locally (no inference API needed).

Downloads files from huggingface.co (plain HTTPS) into ./models/hf_model.
Only huggingface_hub is needed to download. torch/transformers are only
required at inference time by main.py.

Usage:
    python -X utf8 fetch_model_local.py
"""

import os, sys, subprocess, io

# ── UTF-8 safe output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HF_REPO_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
LOCAL_DIR  = "./models/hf_model"
ENV_FILE   = ".env"


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def update_env(key, value):
    lines, found = [], False
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            lines = f.readlines()
        for i, ln in enumerate(lines):
            if ln.startswith(f"{key}=") or ln.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)
    print(f"   >> {key} = {value}  (saved to {ENV_FILE})")


def read_token():
    """Read HF_API_TOKEN from .env."""
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for ln in f:
                if ln.startswith("HF_API_TOKEN="):
                    return ln.split("=", 1)[1].strip()
    return ""


def check_torch():
    """Return True if torch can actually be imported (DLL loaded ok)."""
    try:
        import importlib
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return False
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def check_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def check_tensorflow():
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("IMARA AI -- Download model for 100% local inference")
    print("=" * 60)
    print()
    print(f"  Repo  : {HF_REPO_ID}")
    print(f"  Target: {os.path.abspath(LOCAL_DIR)}")
    print()

    # ── Step 1: ensure huggingface_hub ───────────────────────────────────────
    print("[1/4] Checking huggingface_hub ...")
    try:
        import huggingface_hub  # noqa: F401
        print("      OK -- already installed")
    except ImportError:
        print("      Installing huggingface_hub ...")
        pip("huggingface_hub")

    # ── Step 2: read / ask for token ─────────────────────────────────────────
    print()
    print("[2/4] HuggingFace token ...")
    token = read_token()
    if token:
        print(f"      Found in .env: {token[:10]}...")
    else:
        print("      Not found in .env.")
        print("      Get a free read-token at: https://huggingface.co/settings/tokens")
        token = input("      Paste token (hf_...): ").strip()
        if not token.startswith("hf_"):
            print("      ERROR: token must start with 'hf_'. Aborting.")
            sys.exit(1)
        update_env("HF_API_TOKEN", token)

    # ── Step 3: download model snapshot ──────────────────────────────────────
    print()
    print("[3/4] Downloading model snapshot ...")
    print("      Source : https://huggingface.co")
    print("      This can take 2-10 minutes depending on your connection.")
    print()

    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=LOCAL_DIR,
            token=token,
            # skip large framework-specific blobs we don't need
            ignore_patterns=["*.msgpack", "flax_model*", "rust_model*", "onnx*"],
        )
        print()
        print(f"      Downloaded to: {os.path.abspath(local_path)}")
    except Exception as exc:
        print()
        print(f"      ERROR: {exc}")
        print()
        print("  Possible causes:")
        print("  1. huggingface.co is also blocked on your network (try a VPN/hotspot)")
        print("  2. Token is invalid (get a new one at huggingface.co/settings/tokens)")
        print("  3. Disk space low (model is ~80 MB)")
        sys.exit(1)

    # ── Step 4: choose inference backend ─────────────────────────────────────
    print()
    print("[4/4] Choosing local inference backend ...")

    torch_ok = check_torch()
    tf_ok    = check_tensorflow()
    tr_ok    = check_transformers()

    if torch_ok and tr_ok:
        print("      PyTorch + Transformers -- OK.  Using local PyTorch mode.")
        update_env("USE_HF_API",    "false")
        update_env("HF_MODEL_PATH", LOCAL_DIR)
        backend = "pytorch"

    elif torch_ok and not tr_ok:
        print("      PyTorch found but transformers missing -- installing ...")
        pip("transformers")
        update_env("USE_HF_API",    "false")
        update_env("HF_MODEL_PATH", LOCAL_DIR)
        backend = "pytorch"

    elif not torch_ok and tf_ok:
        print("      PyTorch has a DLL error but TensorFlow is available.")
        print("      WARN: the downloaded HF model needs PyTorch to load.")
        print("      We will reinstall torch (CPU-only) to fix the DLL.")
        print("      Reinstalling torch CPU ...")
        pip("--force-reinstall", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--extra-index-url", "https://pypi.org/simple")
        if check_torch():
            print("      torch reinstalled OK.")
            if not check_transformers():
                pip("transformers")
            update_env("USE_HF_API",    "false")
            update_env("HF_MODEL_PATH", LOCAL_DIR)
            backend = "pytorch"
        else:
            print("      torch still broken after reinstall.")
            print("      Falling back: service will run in demo mode.")
            update_env("USE_HF_API",    "false")
            update_env("HF_MODEL_PATH", LOCAL_DIR)
            backend = "broken_torch"

    else:
        print("      Neither PyTorch nor TensorFlow available.")
        print("      Installing torch CPU ...")
        pip("torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--extra-index-url", "https://pypi.org/simple")
        pip("transformers")
        if check_torch():
            update_env("USE_HF_API",    "false")
            update_env("HF_MODEL_PATH", LOCAL_DIR)
            backend = "pytorch"
        else:
            print("      torch still could not load.  See note below.")
            update_env("USE_HF_API",    "false")
            update_env("HF_MODEL_PATH", LOCAL_DIR)
            backend = "broken_torch"

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if backend == "pytorch":
        print("SUCCESS -- Model ready for local inference.")
        print()
        print("  Start the service:")
        print("      python main.py")
        print()
        print("  The service runs 100% offline -- no internet at inference time.")
    else:
        print("PARTIAL -- Model files downloaded but torch DLL is broken.")
        print()
        print("  To fix the torch DLL issue (common on Windows), try:")
        print("  1. Install the Visual C++ Redistributable:")
        print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("  2. Then run:  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu")
        print("  3. Then run:  python main.py")
        print()
        print("  The service will run in demo mode until torch works.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
