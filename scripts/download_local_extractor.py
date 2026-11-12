"""Download the pinned public model; verify weights before making them usable.

Bounded range requests avoid long stalled single-stream downloads on Windows.
Cache files stay outside Git. No Hugging Face token is required.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import hf_hub_download
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args()
    config = json.loads((ROOT/"experiments/v3/protocol.json").read_text(encoding="utf8"))["extraction"]
    folder = ROOT/"outputs/research_v3/local_model"; folder.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{config['model']}/resolve/{config['revision']}/model.safetensors"
    if args.mirror == "modelscope":
        url = f"https://modelscope.cn/models/{config['model']}/resolve/master/model.safetensors"
    # Resolve immutable weight size and hash from the official model repository.
    info = requests.get(f"https://huggingface.co/api/models/{config['model']}/revision/{config['revision']}", params={"blobs": "true"}, timeout=30)
    info.raise_for_status()
    metadata = next(x for x in info.json()["siblings"] if x["rfilename"] == "model.safetensors")["lfs"]
    size, expected = metadata["size"], metadata["sha256"]
    target = folder/"model.safetensors"
    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    if not target.exists() or digest(target) != expected:
        parts = folder/"parts1m"; parts.mkdir(exist_ok=True)
        chunk = 1024 * 1024
        def get_part(index):
            start = index * chunk; end = min(size, start+chunk)-1
            path = parts/f"{index:05d}.part"
            if path.exists() and path.stat().st_size == end-start+1:
                return
            for attempt in range(5):
                try:
                    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 45))
                    response.raise_for_status()
                    assert response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {start}-{end}/{size}"
                    assert len(response.content) == end-start+1
                    path.write_bytes(response.content)
                    return
                except (requests.RequestException, AssertionError):
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        count = (size+chunk-1)//chunk
        with ThreadPoolExecutor(max_workers=32) as pool:
            tasks = [pool.submit(get_part, i) for i in range(count)]
            for done, task in enumerate(as_completed(tasks), 1):
                task.result()
                if done % 100 == 0:
                    print(f"Verified-size parts {done}/{count}", flush=True)
        temporary = folder/"model.safetensors.partial"
        with temporary.open("wb") as output:
            for i in range(count):
                with (parts/f"{i:05d}.part").open("rb") as source:
                    shutil.copyfileobj(source, output)
        assert temporary.stat().st_size == size and digest(temporary) == expected, "Weight checksum mismatch"
        temporary.replace(target)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE", "README.md"):
        source = hf_hub_download(config["model"], name, revision=config["revision"])
        shutil.copyfile(source, folder/name)
    (folder/"verified_model.json").write_text(json.dumps({"model": config["model"], "revision": config["revision"], "weights_sha256": expected, "bytes": size,
          "weight_download_url": url, "verification": "Full file SHA256 must match the immutable Hugging Face revision, including when using the author's ModelScope mirror"}, indent=2), encoding="utf8")
    print("Local model verified:", folder, flush=True)


if __name__ == "__main__":
    main()
