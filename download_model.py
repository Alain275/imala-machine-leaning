"""
IMARA AI - Model Setup Script
Supports three modes:
  1. Hugging Face Inference API  (cloud, no local model needed — RECOMMENDED)
  2. Create local test model      (needs tensorflow)
  3. Manual download instructions
"""

import os
import sys

HF_REPO_ID  = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
ENV_FILE    = ".env"

# ─────────────────────────────────────────────────────────────────────────────
def setup_hf_api():
    """Configure the app to use the Hugging Face Inference API (no local model)."""
    print("=" * 60)
    print("🌐 HuggingFace Inference API Setup")
    print("=" * 60)
    print()
    print("This mode sends images to HuggingFace's free inference servers.")
    print("✅ No local model download  ✅ No torch / tensorflow needed")
    print()
    print("You need a FREE HuggingFace account and a read token:")
    print("  1. Sign up / log in at https://huggingface.co")
    print("  2. Go to  https://huggingface.co/settings/tokens")
    print("  3. Click  'New token'  →  Role: Read  →  Copy the token")
    print()

    token = input("Paste your HuggingFace token here (hf_...): ").strip()
    if not token.startswith("hf_"):
        print("⚠️  Token should start with 'hf_'. Check and try again.")
        return False

    # ── Quick connectivity test ───────────────────────────────────────────────
    print()
    print("🔄 Testing connectivity to HuggingFace API...")
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{HF_REPO_ID}",
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            body = _json.loads(r.read())
        print("✅ API connection successful!")
        print(f"   Model status: {body.get('pipeline_tag', 'image-classification')}")
    except Exception as e:
        print(f"⚠️  Could not reach API ({e})")
        print("   Your token may be wrong, or HuggingFace is temporarily unavailable.")
        cont = input("   Save token anyway and continue? (y/n): ").strip().lower()
        if cont != "y":
            return False

    # ── Write / update .env ───────────────────────────────────────────────────
    _update_env("HF_API_TOKEN", token)
    _update_env("USE_HF_API", "true")
    _update_env("HF_REPO_ID", HF_REPO_ID)

    print()
    print("=" * 60)
    print("✨ Setup complete!  Start the service with:")
    print("   python main.py")
    print("=" * 60)
    return True


def create_dummy_model():
    """Create a local MobileNetV2 model structure (untrained) for testing."""
    print()
    print("📦 Checking / installing TensorFlow...")
    ret = os.system("pip install tensorflow pillow -q")

    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        print("🔨 Building model structure (ImageNet weights, 38-class head)...")

        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(1024, activation="relu")(x)
        out = Dense(38, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=out)

        os.makedirs("models", exist_ok=True)
        save_path = "./models/plantvillage_mobilenetv2.h5"
        model.save(save_path)

        _update_env("USE_HF_API", "false")

        print(f"✅ Test model saved to {os.path.abspath(save_path)}")
        print()
        print("⚠️  NOTE: weights are random — not trained on plant diseases.")
        print("   Use Option 1 (HF API) for real predictions.")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("TensorFlow could not be installed or loaded.")
        print("Use Option 1 (HF Inference API) which needs zero local ML libraries.")
        return False


def show_manual_instructions():
    print()
    print("=" * 60)
    print("📥 MANUAL OPTIONS")
    print("=" * 60)
    print()
    print("Option A — HuggingFace Inference API (easiest):")
    print("  • No download needed — model runs on HF servers")
    print("  • Just get a free token at https://huggingface.co/settings/tokens")
    print("  • Add to your .env file:")
    print("      USE_HF_API=true")
    print(f"     HF_API_TOKEN=hf_your_token_here")
    print(f"     HF_REPO_ID={HF_REPO_ID}")
    print()
    print("Option B — Kaggle trained .h5 model (local):")
    print("  • https://www.kaggle.com/code/marquis03/plant-disease-classification-using-mobilenetv2")
    print("  • Download output, save as:  models/plantvillage_mobilenetv2.h5")
    print()
    print("Option C — Google search:")
    print("  • Search: 'PlantVillage MobileNetV2 h5 keras model download'")
    print("  • Save to: models/plantvillage_mobilenetv2.h5")
    print()
    print("=" * 60)


def _update_env(key: str, value: str):
    """Write or update a key=value line in the .env file."""
    lines = []
    found = False
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)
    print(f"   ✏️  {key} saved to {ENV_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("🌱 IMARA AI - Plant Disease Detection Setup")
    print("=" * 60)
    print()
    print("Choose setup method:")
    print()
    print("1. 🌐 HuggingFace Inference API  [RECOMMENDED]")
    print("   • No local model / no torch / no tensorflow needed")
    print("   • Real 95% accuracy predictions via HF cloud")
    print("   • Needs a free HuggingFace account + read token")
    print()
    print("2. 🔨 Create local test model  (needs tensorflow)")
    print("   • Random weights — good for integration testing only")
    print()
    print("3. 📋 Manual download instructions")
    print()

    choice = input("Enter choice (1/2/3): ").strip()
    print()

    if choice == "1":
        setup_hf_api()
    elif choice == "2":
        print("⚠️  This creates a model with RANDOM weights — not suitable for real use.")
        if input("Continue? (y/n): ").strip().lower() == "y":
            create_dummy_model()
        else:
            print("Cancelled.")
    elif choice == "3":
        show_manual_instructions()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
