import os
import json
import time
import threading
from typing import Dict, Any, List, Optional


class MetricsManager:
    """Lightweight in-memory metrics with JSONL logging."""

    def __init__(self, log_dir: str = "logs", keep_recent: int = 50):
        self.log_dir = log_dir
        self.keep_recent = keep_recent
        os.makedirs(self.log_dir, exist_ok=True)

        # counters
        self.start_time = time.time()
        self.documents_uploaded = 0
        self.total_pages = 0
        self.total_chunks = 0
        self.total_queries = 0
        self.error_count = 0

        # rolling windows
        self.recent_uploads: List[Dict[str, Any]] = []
        self.recent_queries: List[Dict[str, Any]] = []

        # latency tracking (ms)
        self.upload_latencies: List[float] = []
        self.query_latencies: List[float] = []

        # thread safety
        self._lock = threading.Lock()

    def _append_rolling(self, coll: List[Dict[str, Any]], item: Dict[str, Any]):
        coll.append(item)
        if len(coll) > self.keep_recent:
            del coll[: len(coll) - self.keep_recent]

    def record_upload(
        self,
        file_name: str,
        pages: int,
        chunks: int,
        durations_ms: Dict[str, float],
        status: str = "success",
    ):
        with self._lock:
            ts = time.time()
            total_ms = durations_ms.get("total_ms") or sum(durations_ms.values())
            self.documents_uploaded += 1 if status == "success" else 0
            self.total_pages += pages
            self.total_chunks += chunks
            if status != "success":
                self.error_count += 1

            self.upload_latencies.append(total_ms)
            self._append_rolling(
                self.recent_uploads,
                {
                    "ts": ts,
                    "file": file_name,
                    "pages": pages,
                    "chunks": chunks,
                    "durations_ms": durations_ms,
                    "status": status,
                },
            )

        self._log_event(
            {
                "type": "upload",
                "ts": ts,
                "file": file_name,
                "pages": pages,
                "chunks": chunks,
                "durations_ms": durations_ms,
                "status": status,
            }
        )

    def record_query(
        self,
        question: str,
        success: bool,
        durations_ms: Dict[str, float],
        confidence: Optional[float] = None,
        sources_count: Optional[int] = None,
        model: Optional[str] = None,
        answer_chars: Optional[int] = None,
    ):
        with self._lock:
            ts = time.time()
            total_ms = durations_ms.get("total_ms") or sum(durations_ms.values())
            self.total_queries += 1
            if not success:
                self.error_count += 1
            self.query_latencies.append(total_ms)
            self._append_rolling(
                self.recent_queries,
                {
                    "ts": ts,
                    "question": question,
                    "success": success,
                    "durations_ms": durations_ms,
                    "confidence": confidence,
                    "sources_count": sources_count,
                    "model": model,
                    "answer_chars": answer_chars,
                },
            )

        self._log_event(
            {
                "type": "query",
                "ts": ts,
                "question": question,
                "success": success,
                "durations_ms": durations_ms,
                "confidence": confidence,
                "sources_count": sources_count,
                "model": model,
                "answer_chars": answer_chars,
            }
        )

    def _log_event(self, event: Dict[str, Any]):
        try:
            path = os.path.join(self.log_dir, "metrics.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _avg(self, arr: List[float]) -> float:
        return round(sum(arr) / len(arr), 2) if arr else 0.0

    def _p95(self, arr: List[float]) -> float:
        if not arr:
            return 0.0
        data = sorted(arr)
        k = int(0.95 * (len(data) - 1))
        return round(data[k], 2)

    def system_metrics(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            import torch  # type: ignore

            info.update(
                {
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
            )
        except Exception:
            info.update({"gpu_available": False})

        # Optional psutil metrics
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            info.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "mem_total_gb": round(vm.total / (1024 ** 3), 2),
                    "mem_used_gb": round(vm.used / (1024 ** 3), 2),
                    "mem_percent": vm.percent,
                }
            )
        except Exception:
            pass
        return info

    def summary(self) -> Dict[str, Any]:
        up_ms = self.upload_latencies
        q_ms = self.query_latencies
        uptime_sec = int(time.time() - self.start_time)
        return {
            "uptime_sec": uptime_sec,
            "documents_uploaded": self.documents_uploaded,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_queries": self.total_queries,
            "error_count": self.error_count,
            "avg_upload_ms": self._avg(up_ms),
            "p95_upload_ms": self._p95(up_ms),
            "avg_query_ms": self._avg(q_ms),
            "p95_query_ms": self._p95(q_ms),
            "system": self.system_metrics(),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "recent_uploads": self.recent_uploads,
            "recent_queries": self.recent_queries,
        }
