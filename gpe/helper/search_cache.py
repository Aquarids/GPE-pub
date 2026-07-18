import hashlib
import json
import os
import shutil
import time
from pathlib import Path


class SearchCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def get(self, namespace, params, ttl=None):
        path = self._path(namespace, params)
        if not path.exists():
            return None
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace, params, value):
        path = self._path(namespace, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
        tmp_path.replace(path)

    def clear(self, namespace=None):
        target = self.cache_dir if namespace is None else self.cache_dir / namespace
        if target.exists():
            shutil.rmtree(target)

    def _path(self, namespace, params):
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()
        return self.cache_dir / namespace / f"{digest}.json"
