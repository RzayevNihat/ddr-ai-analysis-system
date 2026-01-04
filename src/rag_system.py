import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    DDR RAG:
    - ChromaDB (persistent) + SentenceTransformer embeddings
    - Groq LLM (LLMService)
    - setup_data run etmədən də işləsin deyə processed_ddrs.json fallback istifadə olunur.
    """

    # Groq 413 riskini azaltmaq üçün sərt limitlər
    MAX_CONTEXT_CHARS_PER_DOC = 1400
    MAX_DOCS_IN_CONTEXT = 3
    MAX_PROMPT_CHARS = 9000  # ümumi prompt limitini də qoruyuruq

    def __init__(self):
        # LLM
        self.llm = LLMService()

        # Embeddings model (Config-də adı: EMBEDDINGS_MODEL)
        self.embedding_model = SentenceTransformer(Config.EMBEDDINGS_MODEL)

        # ChromaDB client (persistent) — Settings eyni saxlanılır (chroma settings conflict riskini azaldır)
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_PERSIST_DIR))

        # Collection
        self.collection = self.client.get_or_create_collection(
            name="ddr_reports",
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding cache (disk)
        self.cache_dir: Path = Path(Config.PROCESSED_DATA_PATH) / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # processed_ddrs.json cache (setup_data run etmədən işləmək üçün)
        self._processed_cache: Optional[List[Dict[str, Any]]] = None

        logger.info("✅ RAG System initialized")

    # -----------------------------
    # Helpers: JSON fallback cache
    # -----------------------------
    def _load_processed_cache(self) -> List[Dict[str, Any]]:
        if self._processed_cache is not None:
            return self._processed_cache

        p = Path(Config.PROCESSED_DATA_PATH) / "processed_ddrs.json"
        if not p.exists():
            self._processed_cache = []
            return self._processed_cache

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = []

        # yalnız düzgün dict olanları saxla
        self._processed_cache = [
            d for d in (data or [])
            if isinstance(d, dict) and "error" not in d
        ]
        return self._processed_cache

    # -----------------------------
    # Helpers: Embedding cache
    # -----------------------------
    def _cache_key(self, text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        key = self._cache_key(text)
        f = self.cache_dir / f"{key}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                return None
        return None

    def _save_embedding(self, text: str, emb: List[float]) -> None:
        key = self._cache_key(text)
        f = self.cache_dir / f"{key}.pkl"
        try:
            with open(f, "wb") as fh:
                pickle.dump(emb, fh)
        except Exception:
            pass

    def _embed(self, text: str) -> List[float]:
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached
        emb = self.embedding_model.encode([text])[0].tolist()
        self._save_embedding(text, emb)
        return emb

    # -----------------------------
    # Text normalize & compacting
    # -----------------------------
    def _normalize_text(self, s: str) -> str:
        s = (s or "").replace("\x00", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _safe_clip(self, s: str, limit: int) -> str:
        s = s or ""
        if len(s) <= limit:
            return s
        return s[: limit - 20] + " ...[kəsildi]..."

    # -----------------------------
    # Build doc text for indexing
    # -----------------------------
    def _create_document_text(self, ddr: Dict[str, Any]) -> str:
        parts: List[str] = []
        parts.append(f"Wellbore: {ddr.get('wellbore', '')}")
        parts.append(f"Period: {ddr.get('period', '')}")
        parts.append(f"Operator: {ddr.get('operator', '')}")
        parts.append(f"Rig: {ddr.get('rig_name', '')}")
        parts.append(f"Depth_MD: {ddr.get('depth_md', '')}")
        parts.append(f"Hole size: {ddr.get('hole_size', '')}")

        summary = ddr.get("summary", {}) or {}
        if summary.get("activities_24h"):
            parts.append(f"\nActivities_24h: {summary.get('activities_24h')}")
        if summary.get("planned_24h"):
            parts.append(f"Planned_24h: {summary.get('planned_24h')}")

        ai_sum = ddr.get("ai_summary", "") or ""
        if ai_sum:
            parts.append(f"\nAI_Summary: {ai_sum}")

        raw = ddr.get("raw_text", "") or ""
        if raw:
            raw = self._normalize_text(raw)
            # indeks üçün raw_text-in bir hissəsi yetərlidir
            parts.append("\nRAW_TEXT:")
            parts.append(self._safe_clip(raw, 5000))

        return "\n".join(parts).strip()

    # -----------------------------
    # Index into Chroma
    # -----------------------------
    def add_documents(self, ddr_list: List[Dict[str, Any]]) -> None:
        if not ddr_list:
            logger.warning("add_documents(): boş siyahı")
            return

        documents, metadatas, ids, embeddings = [], [], [], []

        for i, ddr in enumerate(ddr_list):
            doc_text = self._create_document_text(ddr)

            filename = ddr.get("filename") or ddr.get("source_file") or f"ddr_{i}"
            doc_id = f"ddr::{filename}"

            meta = {
                "wellbore": ddr.get("wellbore", "") or "",
                "period": ddr.get("period", "") or "",
                "depth_md": str(ddr.get("depth_md", "") or ""),
                "filename": filename,
            }

            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(doc_id)
            embeddings.append(self._embed(doc_text))

        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        logger.info(f"✅ Upserted {len(documents)} documents into vector store")

    # -----------------------------
    # Search
    # -----------------------------
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        q_emb = self._embed(query)

        # ❗ include içində "ids" OLMAZ — ids nəticədə ayrıca gəlir
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = res.get("ids", [[]])[0] or []
        docs_list = res.get("documents", [[]])[0] or []
        metas_list = res.get("metadatas", [[]])[0] or []
        dist_list = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids_list)

        out: List[Dict[str, Any]] = []
        for i in range(len(ids_list)):
            out.append({
                "id": ids_list[i],
                "document": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas_list[i] if i < len(metas_list) else {},
                "distance": dist_list[i] if i < len(dist_list) else None,
            })
        return out

    # -----------------------------
    # Structured / deterministic answers (LLM-siz)
    # -----------------------------
    def get_all_wellbores(self) -> List[str]:
        wells = set()

        # JSON fallback
        for d in self._load_processed_cache():
            w = (d.get("wellbore") or "").strip()
            if w:
                wells.add(w)

        # Chroma metadata
        try:
            got = self.collection.get(include=["metadatas"])
            metas = got.get("metadatas", []) or []
            for m in metas:
                w = (m.get("wellbore") or "").strip()
                if w:
                    wells.add(w)
        except Exception:
            pass

        return sorted(wells)

    def _find_events_in_raw(self, patterns: List[str]) -> List[Dict[str, Any]]:
        data = self._load_processed_cache()
        rows: List[Dict[str, Any]] = []
        rx_list = [re.compile(p, re.IGNORECASE) for p in patterns]

        for d in data:
            raw = d.get("raw_text", "") or ""
            raw_n = self._normalize_text(raw)

            if not raw_n:
                continue

            if any(rx.search(raw_n) for rx in rx_list):
                rows.append({
                    "wellbore": d.get("wellbore", ""),
                    "period": d.get("period", ""),
                    "file": d.get("filename") or d.get("source_file") or "",
                })

        return rows

    def _answer_lost_circulation(self) -> Dict[str, Any]:
        # lost circulation / no returns / no return(s)
        rows = self._find_events_in_raw([
            r"\blost circulation\b",
            r"\bno returns\b",
            r"\bno return\b",
            r"\bloss of returns\b",
        ])

        if not rows:
            return {"answer": "Məlumat tapılmadı. (Məhlul itkisi / no returns qeydi görünmür.)", "sources": []}

        lines = ["Tapılan məhlul itkisi (lost circulation / no returns) uyğun qeydlər:"]
        for r in rows[:120]:
            lines.append(f"- {r['wellbore']} | {r['period']} | {r['file']}")
        return {"answer": "\n".join(lines), "sources": rows}

    def _answer_gas_peaks(self, threshold: float) -> Dict[str, Any]:
        data = self._load_processed_cache()
        rows: List[Dict[str, Any]] = []

        for d in data:
            gas = d.get("gas_readings") or []
            if not isinstance(gas, list) or not gas:
                continue

            for g in gas:
                try:
                    pct = float(g.get("gas_percentage", 0) or 0)
                except Exception:
                    continue
                if pct > threshold:
                    rows.append({
                        "wellbore": d.get("wellbore", ""),
                        "period": d.get("period", ""),
                        "depth": g.get("depth", ""),
                        "gas_%": pct,
                        "c1_ppm": g.get("c1_ppm", ""),
                        "file": d.get("filename") or d.get("source_file") or "",
                    })

        if not rows:
            return {"answer": f"Məlumat tapılmadı. {threshold}% üstü qaz piki yoxdur.", "sources": []}

        rows.sort(key=lambda r: (r["wellbore"], -r["gas_%"]))
        lines = [f"Qaz pikləri > {threshold}% olan intervallar:"]
        for r in rows[:120]:
            lines.append(
                f"- {r['wellbore']} | {r['period']} | dərinlik={r['depth']}m | qaz={r['gas_%']}% | C1={r['c1_ppm']} ppm | {r['file']}"
            )
        return {"answer": "\n".join(lines), "sources": rows}

    # -----------------------------
    # LLM QA (limitli context)
    # -----------------------------
    def _build_context(self, results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Groq 413 olmaması üçün context-i kəsirik.
        """
        sources: List[Dict[str, Any]] = []
        ctx_parts: List[str] = []

        for r in (results or [])[: self.MAX_DOCS_IN_CONTEXT]:
            meta = r.get("metadata") or {}
            doc = self._normalize_text(r.get("document", "") or "")

            title = (
                f"Wellbore: {meta.get('wellbore','')}\n"
                f"Period: {meta.get('period','')}\n"
                f"File: {meta.get('filename','')}\n"
            )

            doc_clip = self._safe_clip(doc, self.MAX_CONTEXT_CHARS_PER_DOC)
            ctx_parts.append(title + doc_clip)

            sources.append({
                "wellbore": meta.get("wellbore", ""),
                "period": meta.get("period", ""),
                "file": meta.get("filename", ""),
                "distance": r.get("distance", None),
            })

        context = "\n\n---\n\n".join(ctx_parts).strip()
        return context, sources

    def answer_question(self, question: str, n_context: int = 4) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            return {"answer": "Zəhmət olmasa sual daxil edin.", "sources": []}

        q_low = q.lower()

        # 1) Quyuların siyahısı
        if ("quyu" in q_low and "ad" in q_low) or ("wellbore" in q_low and "list" in q_low):
            wells = self.get_all_wellbores()
            if wells:
                return {"answer": "Sistemdə olan quyular:\n" + "\n".join([f"{w}" for w in wells]), "sources": []}
            return {"answer": "Heç bir quyu tapılmadı.", "sources": []}

        # 2) Lost circulation
        if "lost circulation" in q_low or "məhlul itkisi" in q_low or "no returns" in q_low:
            return self._answer_lost_circulation()

        # 3) Qaz pikləri > X%
        m = re.search(r"(qaz pik(ləri)?).*(\d+(\.\d+)?)\s*%|\b>\s*(\d+(\.\d+)?)\s*%", q_low)
        if ("qaz" in q_low and "pik" in q_low and ">" in q_low) or m:
            # default 1.2
            threshold = 1.2
            nums = re.findall(r"(\d+(?:\.\d+)?)", q_low)
            if nums:
                try:
                    threshold = float(nums[0])
                except Exception:
                    pass
            return self._answer_gas_peaks(threshold)

        # 4) Ümumi LLM rejimi (RAG)
        try:
            results = self.search(q, n_results=max(1, n_context))
        except Exception as e:
            logger.exception("Search error")
            return {"answer": f"Sualın emalı zamanı xəta (search): {e}", "sources": []}

        context, sources = self._build_context(results)

        prompt = (
            "Sən qazma gündəlik hesabatları (DDR) üzrə analitik assistentsən.\n"
            "CAVABI MÜTLƏQ AZƏRBAYCANCA yaz.\n"
            "Əgər məlumat kontekstdə yoxdursa, uydurma — 'Məlumat tapılmadı' de.\n\n"
            f"KONTEKST:\n{context}\n\n"
            f"SUAl: {q}\n\n"
            "CAVAB (Azərbaycanca, qısa və konkret):"
        )

        # əlavə qoruma: prompt çox böyüməsin
        prompt = self._safe_clip(prompt, self.MAX_PROMPT_CHARS)

        try:
            answer = self.llm.generate_text(
                prompt,
                max_tokens=min(900, int(getattr(Config, "MAX_TOKENS", 4096))),
                temperature=getattr(Config, "TEMPERATURE", 0.1),
            )
        except Exception as e:
            logger.exception("LLM error")
            return {"answer": f"Sualın emalı zamanı xəta (LLM): {e}", "sources": sources}

        return {"answer": (answer or "").strip(), "sources": sources}
