from pathlib import Path
from typing import List, Optional
import json
import os
import glob

CACHE_DIR = Path.home() / ".cas" / "models"

HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"

SOURCES = {
    "huggingface": HF_ENDPOINT,
    "hf-mirror": HF_MIRROR_ENDPOINT,
    "modelscope": MODELSCOPE_ENDPOINT,
}


class ModelManager:
    def __init__(self, cache_dir: Path = CACHE_DIR, source: str = "huggingface"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.source = source
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.models = json.load(f)
        else:
            self.models = {}

    def _save_index(self):
        with open(self.index_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def pull(
        self,
        model_id: str,
        source: Optional[str] = None,
        gguf: bool = False,
        quant: Optional[str] = None,
    ) -> Path:
        src = source or self.source
        safe_name = model_id.replace("/", "__")
        local_path = self.cache_dir / safe_name
        print(f"Pulling {model_id} from {src}...")

        if src == "modelscope":
            return self._pull_modelscope(model_id, local_path, gguf=gguf, quant=quant)
        else:
            return self._pull_huggingface(
                model_id, local_path, src, gguf=gguf, quant=quant
            )

    def _pull_huggingface(
        self,
        model_id: str,
        local_path: Path,
        source: str,
        gguf: bool = False,
        quant: Optional[str] = None,
    ) -> Path:
        try:
            endpoint = SOURCES.get(source, HF_ENDPOINT)
            if source == "hf-mirror":
                os.environ["HF_ENDPOINT"] = endpoint
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

            from huggingface_hub import snapshot_download, list_repo_files

            if gguf:
                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {model_id}")
                print(f"Found GGUF files: {', '.join(gguf_files)}")

                if quant:
                    matched = [f for f in gguf_files if quant.upper() in f.upper()]
                    if not matched:
                        raise ValueError(
                            f"Quantization '{quant}' not found. Available: {', '.join(gguf_files)}"
                        )
                    print(f"Downloading only: {', '.join(matched)}")
                    allow_patterns = matched + ["*.json", "tokenizer*"]
                else:
                    allow_patterns = ["*.gguf", "*.json", "tokenizer*"]
            else:
                allow_patterns = None

            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )

            model_type = "gguf" if gguf else "safetensors"
            extra = {"quant": quant} if quant else {}
            self.models[model_id] = {
                "path": str(local_path),
                "status": "ready",
                "source": source,
                "type": model_type,
                **extra,
            }
            self._save_index()
            print(f"✓ Model saved to {local_path} ({model_type})")
        except ImportError:
            print("⚠ huggingface-hub not installed, running in mock mode")
            local_path.mkdir(parents=True, exist_ok=True)
            self.models[model_id] = {
                "path": str(local_path),
                "status": "mock",
                "source": source,
                "type": "mock",
            }
            self._save_index()
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
        return local_path

    def _pull_modelscope(
        self,
        model_id: str,
        local_path: Path,
        gguf: bool = False,
        quant: Optional[str] = None,
    ) -> Path:
        try:
            from modelscope import snapshot_download

            snapshot_download(model_id, local_dir=local_path)
            model_type = "gguf" if gguf else "safetensors"
            extra = {"quant": quant} if quant else {}
            self.models[model_id] = {
                "path": str(local_path),
                "status": "ready",
                "source": "modelscope",
                "type": model_type,
            }
            self._save_index()
            print(f"✓ Model saved to {local_path}")
        except ImportError:
            print("⚠ modelscope not installed, install with: pip install modelscope")
            local_path.mkdir(parents=True, exist_ok=True)
            self.models[model_id] = {
                "path": str(local_path),
                "status": "mock",
                "source": "modelscope",
                "type": "mock",
            }
            self._save_index()
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
        return local_path

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def load(self, model_id: str, quant: Optional[str] = None):
        if model_id not in self.models:
            raise ValueError(
                f"Model {model_id} not found. Use 'cas pull {model_id}' first."
            )
        info = self.models[model_id]
        if info.get("status") == "mock":
            return self._mock_model(), self._mock_tokenizer()

        path = Path(info["path"])
        model_type = info.get("type", "safetensors")

        if model_type == "gguf":
            target_quant = quant or info.get("quant")
            return self._load_gguf(path, quant=target_quant)
        else:
            return self._load_safetensors(path)

    def _load_gguf(self, path: Path, quant: Optional[str] = None):
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        import torch

        gguf_files = list(path.glob("*.gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files found in {path}")

        if quant:
            matched = [f for f in gguf_files if quant.upper() in f.name.upper()]
            if not matched:
                available = [f.name for f in gguf_files]
                raise ValueError(
                    f"Quant '{quant}' not found. Available: {', '.join(available)}"
                )
            gguf_path = matched[0]
        else:
            priority = [
                "Q8_0",
                "Q6_K",
                "Q5_K_M",
                "Q5_K_S",
                "Q4_K_M",
                "Q4_K_S",
                "Q4_0",
                "Q3_K_M",
                "Q2_K",
            ]
            gguf_path = None
            for q in priority:
                found = [f for f in gguf_files if q.upper() in f.name.upper()]
                if found:
                    gguf_path = found[0]
                    break
            if gguf_path is None:
                gguf_path = gguf_files[0]

        print(f"Loading GGUF model: {gguf_path.name}")

        self._ensure_gguf_config(path, gguf_path)

        config = AutoConfig.from_pretrained(path)
        model_class = AutoModelForCausalLM._model_mapping[type(config)]

        tokenizer_path = self._find_or_download_tokenizer(path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = model_class.from_pretrained(
            str(path),
            gguf_file=gguf_path.name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return model, tokenizer

    def _ensure_gguf_config(self, path: Path, gguf_path: Path):
        config_file = path / "config.json"
        if config_file.exists():
            return

        import gguf
        import struct
        import json

        reader = gguf.GGUFReader(str(gguf_path), "r")

        def get_str(key):
            val = reader.get_field(key)
            if not val or not val.parts:
                return None
            last = val.parts[-1]
            return bytes(last).decode()

        def get_uint32(key):
            val = reader.get_field(key)
            if not val or not val.parts:
                return None
            return int(val.parts[-1][0])

        def get_float32(key):
            val = reader.get_field(key)
            if not val or not val.parts:
                return None
            data = bytes(val.parts[-1])
            return struct.unpack("<f", data)[0]

        arch = get_str("general.architecture")
        arch_cap = arch.capitalize().replace("-", "_")
        config = {
            "model_type": arch,
            "architectures": [f"{arch_cap}ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": get_uint32(f"{arch}.embedding_length"),
            "num_hidden_layers": get_uint32(f"{arch}.block_count"),
            "num_attention_heads": get_uint32(f"{arch}.attention.head_count"),
            "num_key_value_heads": get_uint32(f"{arch}.attention.head_count_kv"),
            "intermediate_size": get_uint32(f"{arch}.feed_forward_length"),
            "max_position_embeddings": get_uint32(f"{arch}.context_length"),
            "rms_norm_eps": get_float32(f"{arch}.attention.layer_norm_rms_epsilon"),
            "rope_theta": get_float32(f"{arch}.rope.freq_base"),
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Generated config.json for {arch}")

    def _find_or_download_tokenizer(self, path: Path) -> Path:
        tokenizer_files = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json"]
        has_tokenizer = any(
            (path / f).exists() and (path / f).stat().st_size > 0
            for f in tokenizer_files
        )
        if has_tokenizer:
            return path

        from huggingface_hub import snapshot_download, list_repo_files

        tokenizer_dir = path / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)

        repo_files = list(path.iterdir())
        model_id_guess = None
        for f in repo_files:
            if f.is_file() and f.suffix == ".gguf":
                name = f.name
                if "Instruct" in name:
                    base = (
                        name.split("-Instruct-")[0]
                        if "-Instruct-" in name
                        else name.split("-GGUF")[0]
                    )
                    model_id_guess = f"Qwen/{base}"
                    break

        if not model_id_guess:
            model_id_guess = "Qwen/Qwen2.5-0.5B"

        print(f"Downloading tokenizer from {model_id_guess}...")
        snapshot_download(
            repo_id=model_id_guess,
            local_dir=tokenizer_dir,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
            ],
        )

        for f in tokenizer_dir.iterdir():
            if f.is_file():
                import shutil

                shutil.copy2(f, path / f.name)

        return path

        from huggingface_hub import snapshot_download

        tokenizer_dir = path / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        print("Downloading tokenizer for GGUF model...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-0.5B",
            local_dir=tokenizer_dir,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
            ],
        )
        return tokenizer_dir

    def _load_safetensors(self, path: Path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return model, tokenizer

    def _mock_model(self):
        class MockModel:
            device = "cpu"

            def generate(self, **kwargs):
                import torch

                return torch.tensor([[0, 1, 2]])

        return MockModel()

    def _mock_tokenizer(self):
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                import torch

                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

            def decode(self, tokens, **kwargs):
                return "This is a mock response from caS."

        return MockTokenizer()
