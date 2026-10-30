#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海岳机电 · 设备点检与维修协同 —— 阶段 1：裸问答原型（RAG）
================================================================================
用途
    验证一件事：从「维修手册 / 历史案例 / 备件替代清单」里，能不能检索到正确依据，
    并且把依据以「可核对的引用」形式交给一线，而不是让模型自由发挥。

不做什么
    没有 Web、没有数据库服务依赖、没有前端。单脚本 + 本地语料 + 命令行。

运行
    1) 建索引：  python rag_proto.py ingest
    2) 问答：    python rag_proto.py ask "M-0912 主轴异响怎么判断是不是轴承磨损"
    3) 评测：    python rag_proto.py eval

环境变量（全部可选，都有默认值）
    LLM_PROVIDER      mock | openai_compat        默认 mock（无网络也能跑通链路）
    LLM_BASE_URL      https://api.deepseek.com/v1
    LLM_API_KEY       为空时自动退化为 mock
    LLM_MODEL         deepseek-chat
    LLM_TIMEOUT       30
    LLM_MAX_RETRIES   2
    EMBED_PROVIDER    local | openai_compat       默认 local（字符 n-gram 哈希，零依赖）
    VECTOR_BACKEND    lite | pg                   默认 lite
    DATABASE_URL      postgresql://user:pwd@host:5432/db   （VECTOR_BACKEND=pg 时必填）
    REFUSE_THRESHOLD  0.30   最高余弦相似度低于此值即拒答
    TOPK              3

设计取舍（三条，写在代码前面，免得后面看的人猜）
    1) 章节边界优先于填满 300–500 token 窗口。本语料单节多在 120–250 token，
       强行合并相邻章节会让引用指向错误的节号，得不偿失。切片器对超长节才做窗口切分。
    2) 表格整块保留。维修手册里「预紧力矩 45±5 N·m」一旦被切断，答案就是错的，
       这比召回率低更危险。
    3) 拒答优先于硬答。相似度不足时宁可说「未找到依据，建议转人工」，
       也不让模型把相邻章节的内容改写成答案。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# 0. 全局配置
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "kb"
EVAL_FILE = BASE_DIR / "eval" / "questions.jsonl"
INDEX_FILE = BASE_DIR / "data" / "index.sqlite3"

CONFIG = {
    # 切片
    "chunk_max_tokens": 500,      # 单块上限
    "chunk_min_tokens": 300,      # 软下限，仅用于判断是否需要合并同节段落
    "chunk_overlap_ratio": 0.12,  # 超长节切分时的重叠比例
    # 检索
    "recall_k": 20,               # 每路召回条数。语料 46 块，取 10 时共用词少的正确块进不了候选池
    "topk": int(os.getenv("TOPK", "3")),
    "rrf_k": 60,                  # RRF 平滑常数，业界常用 60
    # 拒答：两道闸门，任一不满足即转人工
    "refuse_threshold": float(os.getenv("REFUSE_THRESHOLD", "0.20")),   # 置信度
    "min_coverage": float(os.getenv("MIN_COVERAGE", "0.20")),           # 问题实词覆盖度
    # LLM
    "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
    "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "llm_api_key": os.getenv("LLM_API_KEY", ""),
    "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "llm_timeout": float(os.getenv("LLM_TIMEOUT", "30")),
    "llm_max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),
    # 向量
    "embed_provider": os.getenv("EMBED_PROVIDER", "local"),
    "embed_dim": 512,
    "vector_backend": os.getenv("VECTOR_BACKEND", "lite"),
    "database_url": os.getenv("DATABASE_URL", ""),
}

REFUSAL_TEXT = "资料中未找到依据，建议转人工核实。"


def _say(msg: str = "") -> None:
    """统一输出，避免 Windows 控制台中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(msg)


# ---------------------------------------------------------------------------
# 1. 语料加载
#    [阶段2复用] 三类语料的解析逻辑可直接搬到 services/knowledge/loader.py
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """切片单元。字段与架构文档里的 kb_chunks 表一一对应。"""

    chunk_id: str
    source_id: str          # 文档编号 / 案例号 / 备件编码 —— 引用的锚点
    doc_type: str           # manual | case | spare
    equipment_model: str
    version: str
    section_path: str
    text: str
    tokens: int
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def est_tokens(text: str) -> int:
    """
    粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token。
    原型阶段不引入 tiktoken，避免为一个估算值增加依赖；误差在 ±15% 内不影响切片决策。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


# 中文标题形如 "3.2 主轴轴承磨损的判断依据"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
DOC_META_RE = re.compile(r"文档编号[：:]\s*([A-Z0-9\-]+)")
VERSION_RE = re.compile(r"版本[：:]\s*([Vv]?[\d.]+)")
MODEL_RE = re.compile(r"适用型号[：:]\s*([A-Z0-9\-/ ]+)")


def ver_label(version: str, default: str = "未标注") -> str:
    """统一版本号显示：源数据里已经带 V 前缀的（V3.2）不要再补一个 v，否则会出现 vV3.2。"""
    v = (version or "").strip()
    if not v or v in ("-", "—") or v.lower() == "none":
        return default
    return v if v[0] in "vV" else f"v{v}"


def parse_markdown(path: Path) -> list[tuple[str, str, str, str, str]]:
    """
    把 markdown 按标题层级切成 (section_path, text, source_id, version, model) 列表。
    标题行本身保留在每个 section 正文开头，因为标题里往往就带着答案的关键词。
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    source_id = (DOC_META_RE.search(raw) or [None, path.stem])[1]  # type: ignore[index]
    version = (VERSION_RE.search(raw) or [None, "V0"])[1]          # type: ignore[index]
    m = MODEL_RE.search(raw)
    # "适用型号：M-0912 / M-0912A" 只取首个机型，否则整串比对永远无法与问题串匹配
    model = re.split(r"[/、,，]", m.group(1))[0].strip() if m else ""

    sections: list[tuple[str, str, str, str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            path_str = " > ".join(t for _, t in stack) if stack else path.stem
            sections.append((path_str, text, source_id, version, model))

    for line in lines:
        hm = HEADING_RE.match(line)
        if hm:
            flush()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, title)]
            buf = [line]           # 标题行纳入正文，保留上下文
        else:
            buf.append(line)
    flush()
    return sections


def parse_cases(path: Path) -> list[tuple[str, str, str, str, str]]:
    """历史案例 CSV：一行一案例，转成一段带字段名的自然语言，便于 BM25 与向量同时匹配。"""
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            case_id = row.get("case_id", "").strip()
            if not case_id:
                continue
            text = (
                f"历史案例 {case_id}｜设备型号 {row.get('equipment_model','')}"
                f"｜故障类型 {row.get('fault_type','')}"
                f"｜现象 {row.get('phenomenon','')}"
                f"｜原因 {row.get('cause','')}"
                f"｜处置 {row.get('solution','')}"
                f"｜停机 {row.get('downtime_minutes','')} 分钟"
                f"｜来源工单 {row.get('source_wo_no','')}"
                f"｜关闭日期 {row.get('closed_date','')}"
                f"｜记录质量 {row.get('record_quality','')}"
            )
            path_str = f"历史案例 > {case_id}"
            out.append((path_str, text, case_id, row.get("closed_date", ""), row.get("equipment_model", "")))
    return out


def parse_spares(path: Path) -> list[tuple[str, str, str, str, str]]:
    """备件替代清单 CSV：一行一备件，替代关系为空时显式写明「未登记」。"""
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            part_no = row.get("part_no", "").strip()
            if not part_no:
                continue
            subs = row.get("substitutes", "").strip()
            subs_text = subs if subs else "无登记替代型号"
            text = (
                f"备件 {part_no}｜名称 {row.get('name','')}"
                f"｜规格 {row.get('spec','')}"
                f"｜适用机型 {row.get('applies_model','')}"
                f"｜可替代型号 {subs_text}"
                f"｜库存 {row.get('stock','')} 件"
                f"｜采购周期 {row.get('lead_time_days','')} 天"
                f"｜备注 {row.get('note','')}"
            )
            out.append((f"备件替代清单 > {part_no}", text, part_no, "", row.get("applies_model", "")))
    return out


def load_corpus() -> list[Chunk]:
    """扫描 data/kb 下的三类语料，全部转成 (section) 列表再切片。"""
    docs: list[tuple[str, str, str, str, str]] = []
    doc_types: list[str] = []

    for p in sorted(KB_DIR.glob("*.md")):
        secs = parse_markdown(p)
        docs.extend(secs)
        # 安全规程单独成一类：它不绑定某一台设备，回答「作业前必须做什么」时
        # 应当优先于任何一台设备的手册。混在 manual 里会被机型手册的余弦分挤掉。
        stem = p.stem.lower()
        dtype = "safety" if stem.startswith(("safety", "saf_", "loto")) else "manual"
        doc_types.extend([dtype] * len(secs))
    for p in sorted(KB_DIR.glob("*case*.csv")):
        secs = parse_cases(p)
        docs.extend(secs)
        doc_types.extend(["case"] * len(secs))
    for p in sorted(KB_DIR.glob("*part*.csv")):
        secs = parse_spares(p)
        docs.extend(secs)
        doc_types.extend(["spare"] * len(secs))

    chunks: list[Chunk] = []
    for (section_path, text, source_id, version, model), doc_type in zip(docs, doc_types):
        for idx, (sub_text, is_table) in enumerate(split_section(text)):
            cid = hashlib.sha1(f"{source_id}|{section_path}|{idx}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source_id=source_id,
                    doc_type=doc_type,
                    equipment_model=model,
                    version=version,
                    section_path=section_path,
                    text=sub_text,
                    tokens=est_tokens(sub_text),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# 2. 结构化切片
#    [阶段2复用] 整个 split_section() 是阶段 2 的核心模块，建议原样搬走
# ---------------------------------------------------------------------------


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_table_block(lines: list[str], max_tokens: int) -> list[str]:
    """
    表格整块保留；只有超过上限时才按行切，且每一片都复制表头与分隔行。
    少了表头，模型看到的就是一堆没有列名的数字。
    """
    if est_tokens("\n".join(lines)) <= max_tokens or len(lines) <= 3:
        return ["\n".join(lines)]

    header, sep, rows = lines[0], lines[1], lines[2:]
    out, cur = [], []
    for r in rows:
        trial = [header, sep] + cur + [r]
        if cur and est_tokens("\n".join(trial)) > max_tokens:
            out.append("\n".join([header, sep] + cur))
            cur = [r]
        else:
            cur.append(r)
    if cur:
        out.append("\n".join([header, sep] + cur))
    return out


def _split_long_text(text: str, max_tokens: int, overlap_ratio: float) -> list[str]:
    """
    超长段落按句切分并做滑动窗口，窗口之间保留 overlap_ratio 的重叠，
    避免「条件在上一段、结论在下一段」被切断。
    """
    sentences = [s for s in re.split(r"(?<=[。；！？\n])", text) if s.strip()]
    if not sentences:
        return [text]

    out, buf, buf_t = [], [], 0
    overlap_tokens = int(max_tokens * overlap_ratio)

    for sent in sentences:
        t = est_tokens(sent)
        if buf and buf_t + t > max_tokens:
            out.append("".join(buf))
            # 保留尾部若干句作为下一窗口的开头
            tail, tail_t = [], 0
            for s in reversed(buf):
                st = est_tokens(s)
                if tail_t + st > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_t += st
            buf, buf_t = tail, tail_t
        buf.append(sent)
        buf_t += t
    if buf:
        out.append("".join(buf))
    return [c for c in out if c.strip()]


def split_section(text: str) -> list[tuple[str, bool]]:
    """
    把一个 section 拆成若干切片单元，返回 [(文本, 是否含表格)]。

    规则：
      - 连续的表格行 → 一个不可分割的块（超长时按行切并复制表头）
      - 连续的非表格行 → 一个文本块（超长时按句切 + 重叠）
      - 相邻块贪心合并到上限，但不超过 chunk_max_tokens
    """
    max_t = CONFIG["chunk_max_tokens"]
    units: list[tuple[str, bool]] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]):
            j = i
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            for piece in _split_table_block(lines[i:j], max_t):
                units.append((piece, True))
            i = j
        else:
            j = i
            while j < len(lines) and not _is_table_line(lines[j]):
                j += 1
            block = "\n".join(lines[i:j]).strip()
            if block:
                if est_tokens(block) > max_t:
                    for piece in _split_long_text(block, max_t, CONFIG["chunk_overlap_ratio"]):
                        units.append((piece, False))
                else:
                    units.append((block, False))
            i = j

    # 贪心合并：只合并「小文本块」，表格块不与其他块合并，保证引用指向明确
    merged: list[tuple[str, bool]] = []
    buf, buf_t, buf_has_table = "", 0, False
    for piece, is_table in units:
        t = est_tokens(piece)
        if is_table:
            if buf:
                # 只有标题行（如「### 4.1 轴承预紧力矩」）时不单独成块，
                # 直接拼到表格前面 —— 否则会出现 5 个 token 的「标题块」占掉一个召回位，
                # 而且表格片段的文本里没有章节名，检索时非常难命中。
                if _is_heading_only(buf):
                    merged.append((f"{buf.strip()}\n{piece}", True))
                    buf, buf_t, buf_has_table = "", 0, False
                    continue
                merged.append((buf.strip(), buf_has_table))
                buf, buf_t, buf_has_table = "", 0, False
            merged.append((piece, True))
            continue
        if buf and buf_t + t > max_t:
            merged.append((buf.strip(), buf_has_table))
            buf, buf_t, buf_has_table = "", 0, False
        buf = f"{buf}\n\n{piece}" if buf else piece
        buf_t += t
    if buf:
        merged.append((buf.strip(), buf_has_table))
    # 只有标题、没有正文的父级章节（如「## 2 技术参数」下面直接是「### 2.1」）
    # 不进索引：它的信息已经在子章节的 section_path 里，留着只会占召回位。
    return [m for m in merged if not _is_heading_only(m[0])]


def _is_heading_only(text: str) -> bool:
    """判断一个块是不是「只有一行标题」，这类块单独建索引没有检索价值。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) == 1 and lines[0].lstrip().startswith("#")


# ---------------------------------------------------------------------------
# 3. 向量化
#    [阶段2复用] 接口保持不变，本地哈希实现可在没有 embedding 预算时兜底
# ---------------------------------------------------------------------------


class Embedder:
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalHashingEmbedder(Embedder):
    """
    零依赖的字符 n-gram 哈希向量。
    中文没有空格，按字符 bigram 做特征比按词更稳（不依赖词典），代价是语义泛化弱。
    生产环境应换成 bge-m3 / text-embedding-3 之类的模型，接口不变。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _grams(text: str) -> list[tuple[str, float]]:
        t = re.sub(r"\s+", "", text)
        grams: list[tuple[str, float]] = []
        grams.extend((t[i : i + 2], 1.0) for i in range(len(t) - 1))
        grams.extend((ch, 0.35) for ch in t)   # unigram 兜底，提升短词/编号命中
        return grams

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for gram, w in self._grams(text):
                h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "big")
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign * w
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OpenAICompatEmbedder(Embedder):
    """OpenAI 兼容 /embeddings 接口。供应商换 base_url + model 即可。"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx  # 延迟导入：不装 httpx 也能跑 local 模式

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(texts)},
            )
            r.raise_for_status()
            data = r.json()["data"]
        return [d["embedding"] for d in data]


def get_embedder() -> Embedder:
    if CONFIG["embed_provider"] == "openai_compat":
        return OpenAICompatEmbedder(
            CONFIG["llm_base_url"], CONFIG["llm_api_key"],
            os.getenv("EMBED_MODEL", "text-embedding-3-small"), CONFIG["embed_dim"],
        )
    return LocalHashingEmbedder(CONFIG["embed_dim"])


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # 两边都已归一化，点积即余弦


# ---------------------------------------------------------------------------
# 4. 存储层
#    接口统一，pg 与 lite 可互换。架构里最终用 pgvector，lite 只用于本机跑通。
#    [阶段2复用] Store 接口 + LiteStore 的 BM25 实现可直接复用
# ---------------------------------------------------------------------------


class Store:
    def rebuild(self, chunks: list[Chunk]) -> None: ...
    def load(self) -> None: ...
    def all_chunks(self) -> list[Chunk]: ...
    def vector_search(self, qvec: list[float], k: int) -> list[tuple[str, float]]: ...
    def bm25_search(self, terms: list[str], k: int) -> list[tuple[str, float]]: ...
    def term_idf(self, term: str) -> float:
        """词权重。lite 后端用真实 df 计算；pg 后端返回 1.0 退化为等权。"""
        return 1.0


class LiteStore(Store):
    """
    SQLite 持久化 + 内存检索。适合几千个 chunk 的原型阶段。
    BM25 用纯 Python 实现（语料小，性能足够），向量用点积。
    """

    def __init__(self, path: Path):
        self.path = path
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._avgdl = 1.0

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("DROP TABLE IF EXISTS kb_chunks")
            conn.execute("CREATE TABLE kb_chunks (chunk_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO kb_chunks (chunk_id, payload) VALUES (?, ?)",
                [(c.chunk_id, json.dumps(c.to_dict(), ensure_ascii=False)) for c in chunks],
            )
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            raise SystemExit(f"索引不存在：{self.path}，请先运行 `python rag_proto.py ingest`")
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT payload FROM kb_chunks").fetchall()
        self._chunks = {}
        for (payload,) in rows:
            c = Chunk(**json.loads(payload))
            self._chunks[c.chunk_id] = c
        self._tf = {cid: Counter(tokenize(c.text)) for cid, c in self._chunks.items()}
        self._df = Counter()
        for tf in self._tf.values():
            self._df.update(tf.keys())
        total = sum(sum(tf.values()) for tf in self._tf.values())
        self._avgdl = total / max(1, len(self._tf))

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def vector_search(self, qvec: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(cid, cosine(qvec, c.embedding)) for cid, c in self._chunks.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def bm25_search(self, terms: list[str], k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
        n = max(1, len(self._chunks))
        scored = []
        for cid, tf in self._tf.items():
            dl = sum(tf.values())
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / self._avgdl))
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def term_idf(self, term: str) -> float:
        """词的逆文档频率，用于给「覆盖面」加权：越少见的词，命中它越说明找对了。"""
        n = max(1, len(self._chunks))
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))


class PgVectorStore(Store):
    """
    PostgreSQL + pgvector 后端，与目标架构一致。

    注意（务必看）：
      1) 本机没有 PG 实例，这段代码未经运行验证，只保证接口与 LiteStore 对齐。
      2) 中文全文检索用 `simple` 配置时依赖预分词：入库时把 jieba 分词结果用空格拼成
         一份 tsv 文本，否则整段中文会被当成一个 token。生产建议装 zhparser。
      3) HNSW 索引参数在 10 万级 chunk 以下用默认值即可，超出后调 m / ef_construction。
    """

    def __init__(self, dsn: str, dim: int):
        self.dsn = dsn
        self.dim = dim
        self._conn = None

    def _connect(self):
        if self._conn is None:
            import psycopg  # 延迟导入
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def rebuild(self, chunks: list[Chunk]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS kb_chunks")
            cur.execute(f"""
                CREATE TABLE kb_chunks (
                    chunk_id        TEXT PRIMARY KEY,
                    source_id       TEXT NOT NULL,
                    doc_type        TEXT NOT NULL,
                    equipment_model TEXT,
                    version         TEXT,
                    section_path    TEXT,
                    text            TEXT NOT NULL,
                    tokens          INT,
                    embedding       vector({self.dim}),
                    tsv             tsvector
                )""")
            cur.execute("CREATE INDEX kb_chunks_vec_idx ON kb_chunks USING hnsw (embedding vector_cosine_ops)")
            cur.execute("CREATE INDEX kb_chunks_tsv_idx ON kb_chunks USING gin (tsv)")
            for c in chunks:
                tsv_text = " ".join(tokenize(c.text))
                cur.execute(
                    "INSERT INTO kb_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,to_tsvector('simple',%s))",
                    (c.chunk_id, c.source_id, c.doc_type, c.equipment_model, c.version,
                     c.section_path, c.text, c.tokens, c.embedding, tsv_text),
                )

    def load(self) -> None:
        self._connect()

    def all_chunks(self) -> list[Chunk]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id,source_id,doc_type,equipment_model,version,section_path,text,tokens FROM kb_chunks")
            rows = cur.fetchall()
        return [Chunk(*r) for r in rows]

    def vector_search(self, qvec: list[float], k: int) -> list[tuple[str, float]]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score "
                "FROM kb_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (qvec, qvec, k),
            )
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    def bm25_search(self, terms: list[str], k: int) -> list[tuple[str, float]]:
        conn = self._connect()
        query = " | ".join(t.replace("'", "") for t in terms if t.strip())
        if not query:
            return []
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, ts_rank_cd(tsv, to_tsquery('simple', %s)) AS s "
                "FROM kb_chunks WHERE tsv @@ to_tsquery('simple', %s) "
                "ORDER BY s DESC LIMIT %s",
                (query, query, k),
            )
            return [(r[0], float(r[1])) for r in cur.fetchall()]


def build_store() -> Store:
    if CONFIG["vector_backend"] == "pg":
        if not CONFIG["database_url"]:
            raise SystemExit("VECTOR_BACKEND=pg 时必须设置 DATABASE_URL")
        return PgVectorStore(CONFIG["database_url"], CONFIG["embed_dim"])
    return LiteStore(INDEX_FILE)


# ---------------------------------------------------------------------------
# 5. 混合检索：分词 → BM25 + 向量 → RRF 融合 → 轻量重排
#    [阶段2复用] Retriever 是整个链路最值得复用的部分
# ---------------------------------------------------------------------------

_JIEBA_READY = False


def tokenize(text: str) -> list[str]:
    """
    中文分词。优先 jieba（search 模式能切出「主轴轴承」这类复合词），
    没装则退化为字 bigram —— 效果差一些，但不会因为缺依赖跑不起来。
    """
    global _JIEBA_READY
    if not _JIEBA_READY:
        try:
            import jieba
            jieba.setLogLevel(60)      # 关掉 "Building prefix dict" 之类的日志
            _JIEBA_READY = True
        except Exception:
            _JIEBA_READY = False

    if _JIEBA_READY:
        import jieba
        toks = [t.strip() for t in jieba.lcut_for_search(text) if t.strip()]
    else:
        t = re.sub(r"\s+", "", text)
        toks = [t[i : i + 2] for i in range(len(t) - 1)]

    # 过滤纯标点，保留数字与编号（45、6204、M-0912 这类是硬答案）
    return [t for t in toks if re.search(r"[\w\u4e00-\u9fff]", t)]


# 检索用的停用词：不参与词覆盖与 BM25 查询，否则「的/是/多少」这类词会抬高所有片段的分数，
# 让「什么问题都能匹配上」，直接影响拒答判断。
STOPWORDS = {
    "的", "是", "了", "在", "和", "与", "或", "有", "无", "为", "对", "把", "被", "从", "到",
    "时", "后", "前", "中", "上", "下", "个", "们", "我", "你", "他", "它", "这", "那", "哪",
    "多少", "怎么", "怎样", "如何", "什么", "为何", "为什么", "是否", "哪些", "哪个", "一下",
    "需要", "必须", "可以", "应该", "能否", "请问", "以及", "还是", "就是", "如果", "但", "而且",
}


def query_terms(text: str) -> list[str]:
    """面向检索的问题词：分词 + 去停用词 + 去重保序。"""
    seen, out = set(), []
    for t in tokenize(text):
        if t in STOPWORDS or len(t) == 1 and not t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# 领域同义词：一线口头说法与规程用词往往对不上。
# 「必须做哪些安全措施」在规程里写作「停机与挂牌步骤」，字面只共享「主轴」一个泛词，
# BM25 直接落选。这张表只用于召回扩词，不参与覆盖度计算——否则分母被撑大，
# 拒答闸门会跟着失灵。生产环境应换成同义词库或查询改写模型。
SYNONYMS = {
    "安全措施": ("挂牌", "上锁", "LOTO", "隔离", "停机"),
    "安全": ("挂牌", "上锁", "LOTO", "隔离"),
    "措施": ("步骤", "条件", "挂牌"),
    "复产": ("复产确认", "解除挂牌", "试运行"),
    "误报": ("误报处置", "暂停", "核对"),
}


def expand_terms(terms: list[str]) -> list[str]:
    """召回用扩词：命中同义词表就把对应领域词补进去，原有的词一个不动。"""
    extra: list[str] = []
    for t in terms:
        extra.extend(SYNONYMS.get(t, ()))
    seen = set(terms)
    return terms + [t for t in extra if t not in seen]


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60,
             weights: Sequence[float] | None = None) -> dict[str, float]:
    """
    Reciprocal Rank Fusion：只用名次不用分数，天然消除 BM25 与余弦的量纲差异。
    权重可按两路检索器的可靠度调：本原型的向量是字符哈希实现，语义能力弱，
    所以词法通路权重更高（0.65 / 0.35）。换成真正的 embedding 模型后应回调到 0.5/0.5。
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + w * (1.0 / (k + rank))
    return fused


class Retriever:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.by_id = {c.chunk_id: c for c in store.all_chunks()}

    def retrieve(self, question: str, topk: int | None = None, trace: bool = False) -> dict[str, Any]:
        topk = topk or CONFIG["topk"]
        terms = query_terms(question)
        qvec = self.embedder.embed([question])[0]

        # 召回用扩词版，打分与覆盖度仍用原词表，避免同义词把分母撑大影响拒答判断
        bm25 = self.store.bm25_search(expand_terms(terms), CONFIG["recall_k"])
        vec = self.store.vector_search(qvec, CONFIG["recall_k"])

        fused = rrf_fuse([bm25, vec], CONFIG["rrf_k"], weights=[0.65, 0.35])
        cand_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])[: CONFIG["recall_k"]]]

        bm25_map = dict(bm25)
        bm25_max = max(bm25_map.values(), default=1.0) or 1.0
        qterm_set = set(terms)
        # 用 IDF 给覆盖度加权：命中「挂牌」「解除」这类低频词，比命中「主轴」这种满篇都是的词
        # 更能说明找对了片段。等权覆盖会把泛词算得过重，导致拒答判断失灵。
        idf = {t: self.store.term_idf(t) for t in terms}
        idf_total = sum(idf.values()) or 1.0
        asks_history = any(k in question for k in ("上次", "历史", "案例", "以前", "曾经", "那次"))
        asks_spare = any(k in question for k in ("备件", "替代", "库存", "型号", "采购"))
        asks_safety = any(k in question for k in ("安全", "挂牌", "上锁", "LOTO", "隔离", "授权", "措施", "违规"))
        # 问题里点名的机型（M-0912 / H-1023 这类）。用于判断「这段资料是不是这台设备的」
        named_models = set(re.findall(r"[A-Z]{1,2}-\d{4}[A-Z]?", question))

        reranked = []
        for cid in cand_ids:
            chunk = self.by_id[cid]
            cos = cosine(qvec, chunk.embedding)
            chunk_terms = set(tokenize(chunk.text))
            # 词覆盖：问题里的实词有多少被这个片段覆盖，是「有没有依据」最直接的信号
            matched = qterm_set & chunk_terms
            coverage = sum(idf[t] for t in matched) / idf_total
            bm_norm = bm25_map.get(cid, 0.0) / bm25_max
            # 机型匹配加成：问 M-0912 时优先给 M-0912 的资料，避免串机型给出错误参数
            model_boost = 0.05 if (chunk.equipment_model and chunk.equipment_model in question) else 0.0
            # 跨机型惩罚：问 M-0912 却召回 H-1023 的段落，是比「排得靠后」更严重的错误——
            # 它会把另一台设备的要求当作本机的依据写进答案。判断没指定机型或资料不绑定机型时不惩罚。
            if named_models and chunk.equipment_model and chunk.equipment_model not in named_models:
                model_boost = -0.06
            # 文档类型先验：问「怎么判断/标准是什么」时，手册是规范，案例只是先例；
            # 问「上次/历史怎么处理」时反过来。没有这一层，案例常把手册挤下去。
            prio = 0.0
            if asks_history and chunk.doc_type == "case":
                prio = 0.06
            elif not asks_history and chunk.doc_type == "manual":
                prio = 0.04
            if asks_spare and chunk.doc_type == "spare":
                prio = 0.06
            if asks_safety and chunk.doc_type == "safety":
                prio = 0.10
            score = 0.40 * cos + 0.35 * coverage + 0.25 * bm_norm + model_boost + prio
            reranked.append((chunk, score, cos, coverage, bm_norm))

        reranked.sort(key=lambda x: -x[1])
        top = reranked[:topk]
        # 置信度与排序分分开算：排序要兼顾 BM25 名次，置信度只回答一个问题——
        # 「这段资料里，问题真正想问的东西到底有没有」。
        confidence = (0.45 * top[0][2] + 0.55 * top[0][3]) if top else 0.0

        result = {
            "question": question,
            "terms": terms,
            "top": top,
            "max_cos": top[0][2] if top else 0.0,
            "confidence": confidence,
            "coverage": top[0][3] if top else 0.0,
        }
        if trace:
            _say(f"\n[检索] 检索词：{' / '.join(terms)}")
            _say(f"[检索] BM25 命中 {len(bm25)} 条，向量命中 {len(vec)} 条，融合后候选 {len(cand_ids)} 条")
            for i, (chunk, score, cos, cov, bmn) in enumerate(top, 1):
                _say(f"  {i}. [{chunk.source_id} §{chunk.section_path}] score={score:.3f} "
                     f"(cos={cos:.3f}, cov={cov:.2f}, bm25={bmn:.2f}, {chunk.tokens}t)")
        return result


# ---------------------------------------------------------------------------
# 6. LLM 封装：供应商可替换
#    [阶段2复用] 换供应商只改 base_url/model，业务代码不动
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    provider: str


class LLMClient:
    """
    统一入口。子类只需实现 _generate()。
    记录 tokens 与延迟不是「顺便加个日志」，而是模型可替换的前提：
    不埋这个点，换供应商只能靠感觉。
    """

    name = "base"

    def _generate(self, messages: list[dict[str, str]], context_chunks: list[Any]) -> tuple[str, int, int]:
        raise NotImplementedError

    def complete(self, messages: list[dict[str, str]], context_chunks: list[Any] | None = None) -> LLMResult:
        t0 = time.time()
        text, tin, tout = self._generate(messages, context_chunks or [])
        return LLMResult(
            text=text.strip(),
            model=CONFIG["llm_model"] if self.name != "mock" else "mock-extractive",
            tokens_in=tin,
            tokens_out=tout,
            latency_ms=int((time.time() - t0) * 1000),
            provider=self.name,
        )


# 结构化案例行里，有信息量的字段与可丢弃的元数据字段
_CASE_KEEP = ("故障类型", "现象", "原因", "处置", "停机", "判断依据", "建议", "预紧力矩", "阈值")
_CASE_DROP = ("来源工单", "关闭日期", "记录质量", "设备型号", "历史案例", "设备编号")


def _mock_units(body: str) -> list[str]:
    """
    把切片原文压成「人话」，供抽取式 mock 使用。

    为什么不能直接按标点切：案例行是 `字段 值｜字段 值` 的扁平行，
    按「；」切会把「现象」和它的「原因/处置」切散，最后抽出来的往往是
    「来源工单 xx｜记录质量 完整」这种没有信息量的碎片 —— 答案看着有引用，
    实际问题根本没被回答。三类形态分别处理。
    """
    body = body.replace("\r", "")

    # ① 结构化案例行：字段 值｜字段 值 → 只保留有信息量的字段，故障类型提到句首
    if "｜" in body:
        head: list[str] = []
        for part in (p.strip() for p in body.split("｜")):
            if not part:
                continue
            field, _, val = part.partition(" ")
            if field in _CASE_DROP:
                continue
            if field == "故障类型":
                head.insert(0, val)
            elif field in _CASE_KEEP:
                head.append(f"{field} {val}")
        if head:
            return ["；".join(head)]

    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

    # ② markdown 表格：表头 + 数据行 → 「主体：字段=值，字段=值」，表格结构不会被打散
    table = [ln for ln in lines if ln.startswith("|")]
    if len(table) >= 2:
        def _cells(ln: str) -> list[str]:
            return [c.strip() for c in ln.strip("|").split("|")]

        header = _cells(table[0])
        out: list[str] = []
        for ln in table[1:]:
            if set(ln) <= set("|-: "):      # 分隔行
                continue
            row = _cells(ln)
            if len(row) != len(header):
                continue
            rest = [f"{header[i]}={row[i]}" for i in range(1, len(row)) if row[i]]
            if rest:
                out.append(f"{row[0]}：" + "，".join(rest))
        if out:
            return out

    # ③ 普通段落：按句切，剔除标题行（章节名已在引用的 § 里，正文再说一遍是噪音）
    prose = "\n".join(ln for ln in lines if not ln.startswith("#"))
    sents = [s.strip() for s in re.split(r"(?<=[。；！？\n])", prose) if len(s.strip()) > 8]
    return [s for s in sents if not s.startswith("文档编号")]


class MockProvider(LLMClient):
    """
    无网络 / 无 key 时的确定性生成器：从提示词里的【资料 N】块中抽取与问题最相关的句子，
    并原样附上引用。它不产生任何未出现在资料里的内容，用来跑通链路与评测。
    真实效果必须用真实模型重跑 —— 这一点在评测输出里会明确标注。
    """

    name = "mock"

    def _generate(self, messages, context_chunks):
        user_msg = messages[-1]["content"]
        question = user_msg.split("【问题】")[-1].strip()

        blocks = re.findall(
            r"【资料\s*(\d+)】文档编号：(.+?)｜章节：(.+?)｜版本：(.+?)\n内容：(.*?)(?=\n?【资料|\n?【问题】|$)",
            user_msg, re.S,
        )
        if not blocks:
            return REFUSAL_TEXT, 0, 0

        qterms = set(tokenize(question))
        picked: list[str] = []
        for _idx, src, sec, ver, body in blocks:
            # 凑够 3 条就停：手册一段能出 2 条，案例一条记录只出 1 条，
            # 按「条数」而不是「块数」控制，案例题才不会被前两块占满而漏掉最相关的第 3 条。
            if len(picked) >= 3:
                break
            # 结构化内容（案例行/表格）整块作为一条结论，不再按标点切碎
            units = _mock_units(body)
            if not units:
                continue
            scored = sorted(
                units,
                key=lambda s: len(qterms & set(tokenize(s))),
                reverse=True,
            )
            best = [s for s in scored[:2] if qterms & set(tokenize(s))] or scored[:1]
            for s in best[: 3 - len(picked)]:
                # 结尾的冒号/竖线也要剥掉，否则会出现「…如下：。」这种叠标点
                picked.append(f"{s.rstrip('。；;，,：:｜|')}。[来源: {src} §{sec} | {ver}]")

        if not picked:
            return REFUSAL_TEXT, 0, 0
        body = "\n".join(f"- {p}" for p in picked[:3])
        return f"依据资料，结论如下：\n{body}", est_tokens(user_msg), est_tokens(body)


class OpenAICompatProvider(LLMClient):
    """OpenAI 兼容 /chat/completions。带超时、重试、用量记录。"""

    name = "openai_compat"

    def _generate(self, messages, context_chunks):
        import httpx

        last_err: Exception | None = None
        for attempt in range(CONFIG["llm_max_retries"] + 1):
            try:
                with httpx.Client(timeout=CONFIG["llm_timeout"]) as client:
                    r = client.post(
                        f"{CONFIG['llm_base_url'].rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {CONFIG['llm_api_key']}"},
                        json={
                            "model": CONFIG["llm_model"],
                            "messages": messages,
                            "temperature": 0.1,
                            "max_tokens": 800,
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                usage = data.get("usage", {})
                return (
                    data["choices"][0]["message"]["content"],
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
            except Exception as e:       # noqa: BLE001 —— 原型阶段统一兜底，便于降级
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        return f"[LLM 调用失败，已降级] {last_err}", 0, 0


def get_llm() -> LLMClient:
    if CONFIG["llm_provider"] == "openai_compat" and CONFIG["llm_api_key"]:
        return OpenAICompatProvider()
    return MockProvider()


# ---------------------------------------------------------------------------
# 7. 提示词拼装 + 引用校验
#    [阶段2复用] 引用校验是项目里最不能省的一段：无引用 = 不可用
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是海岳机电的设备点检与维修协同助手，服务对象是维修技师与设备管理人员。

必须遵守：
1. 只能依据【资料】作答，不得使用资料以外的知识补全，不得推测。
2. 每一条结论后面必须紧跟引用，格式严格为：[来源: 文档编号 §章节 | 版本]
3. 参数、力矩、周期、阈值必须原样引用资料中的数值，不得四舍五入或改写单位。
4. 资料不足以回答时，直接输出：资料中未找到依据，建议转人工核实。不得给出笼统建议。
5. 禁止给出解除挂牌、旁路保护、强制复位、跳过安全确认类操作指引；遇到此类请求，提示按安全规程执行并转安全负责人。
6. 涉及判断类问题，必须说明「满足哪几条才能判定」，不能只给结论。
"""


def build_messages(question: str, chunks: list[Any]) -> list[dict[str, str]]:
    blocks = []
    for i, (chunk, _score, _cos, _cov, _bm) in enumerate(chunks, 1):
        blocks.append(
            f"【资料 {i}】文档编号：{chunk.source_id}｜章节：{chunk.section_path}"
            f"｜版本：{chunk.version or '未标注'}\n内容：{chunk.text}"
        )
    user = (
        "以下是检索到的资料，请严格依据资料回答，并逐条标注引用。\n\n"
        + "\n\n".join(blocks)
        + f"\n\n【问题】{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# 引用格式：[来源: 文档编号 §章节 | 版本]，中英文冒号与竖线都容错
CITE_RE = re.compile(r"\[来源\s*[:：]\s*([^§\]\|｜]+?)\s*§\s*([^\|\]｜]+?)\s*[\|｜]\s*([^\]]+?)\s*\]")


def parse_citations(answer: str) -> list[tuple[str, str, str]]:
    return [(a.strip(), b.strip(), c.strip()) for a, b, c in CITE_RE.findall(answer)]


def validate_citations(answer: str, chunks: list[Any]) -> dict[str, Any]:
    """
    校验回答里的每一条引用是否真的指向本次检索到的片段。
    判定标准：
      - 文档编号必须存在于检索结果
      - 章节名必须是该文档某个片段的 section_path 的子串
      - 版本号（若有）必须一致
    不通过的一律计为「编造引用」，这是审计要抓的东西。
    """
    cites = parse_citations(answer)
    valid, invalid = [], []
    for src, sec, ver in cites:
        hit = None
        for chunk, *_ in chunks:
            if chunk.source_id != src:
                continue
            if sec and sec in chunk.section_path:
                if ver and chunk.version and ver not in chunk.version and chunk.version not in ver:
                    continue
                hit = chunk
                break
        (valid if hit else invalid).append((src, sec, ver))
    return {"total": len(cites), "valid": valid, "invalid": invalid}


# ---------------------------------------------------------------------------
# 8. 问答主链路
# ---------------------------------------------------------------------------


def answer(question: str, retriever: Retriever, llm: LLMClient, trace: bool = False) -> dict[str, Any]:
    t0 = time.time()
    r = retriever.retrieve(question, trace=trace)

    # 拒答闸门：最高余弦相似度不足，直接转人工，不调模型
    # 这一步放在调模型之前，省一次 token，也避免模型「硬答」
    if (not r["top"]
            or r["confidence"] < CONFIG["refuse_threshold"]
            or r["coverage"] < CONFIG["min_coverage"]):
        return {
            "question": question,
            "answer": REFUSAL_TEXT,
            "refused": True,
            "reason": (f"confidence={r['confidence']:.3f}（阈值 {CONFIG['refuse_threshold']}）、"
                       f"coverage={r['coverage']:.3f}（阈值 {CONFIG['min_coverage']}）未同时达标"),
            "chunks": r["top"],
            "citations": {"total": 0, "valid": [], "invalid": []},
            "llm": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    messages = build_messages(question, r["top"])
    res = llm.complete(messages, [c for c, *_ in r["top"]])
    cites = validate_citations(res.text, r["top"])

    # 无合法引用 = 不可用，直接降级为拒答（项目红线，不是可选项）
    if cites["total"] == 0 or not cites["valid"]:
        return {
            "question": question,
            "answer": REFUSAL_TEXT,
            "refused": True,
            "reason": "回答未携带有效引用，按红线降级为拒答",
            "chunks": r["top"],
            "citations": cites,
            "llm": res,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    return {
        "question": question,
        "answer": res.text,
        "refused": False,
        "reason": "",
        "chunks": r["top"],
        "citations": cites,
        "llm": res,
        "raw_answer": res.text,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


# ---------------------------------------------------------------------------
# 9. 评测
# ---------------------------------------------------------------------------


def load_questions() -> list[dict[str, Any]]:
    with EVAL_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _gold_hit(chunks: list[Any], gold: list[list[str]]) -> bool:
    for chunk, *_ in chunks:
        for src, sec in gold:
            if chunk.source_id == src and (sec in chunk.section_path or sec in chunk.source_id):
                return True
    return False


def run_eval(retriever: Retriever, llm: LLMClient) -> int:
    qs = load_questions()
    rows = []
    for q in qs:
        out = answer(q["question"], retriever, llm)
        top = out["chunks"]
        gold = q.get("gold", [])
        hit = _gold_hit(top, gold) if gold else False
        cites = out["citations"]
        cite_valid = len(cites["valid"])
        cite_total = cites["total"]
        # 引用的「指向正确率」：合法引用中，有多少指向 gold 片段
        gold_cite = sum(
            1 for src, sec, _v in cites["valid"]
            if any(src == g[0] and (g[1] in sec or sec in g[1] or g[1] in src) for g in gold)
        ) if gold else 0
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "hit": hit,
                "refused": out["refused"],
                "cite_total": cite_total,
                "cite_valid": cite_valid,
                "gold_cite": gold_cite,
                "conf": (0.45 * top[0][2] + 0.55 * top[0][3]) if top else 0.0,
                "cov": top[0][3] if top else 0.0,
                "ms": out["elapsed_ms"],
                "note": q.get("note", ""),
            }
        )

    pos = [r for r in rows if r["type"] == "positive"]
    neg = [r for r in rows if r["type"] == "negative"]

    hit_at_3 = sum(1 for r in pos if r["hit"]) / max(1, len(pos))
    cite_precision = (
        sum(r["cite_valid"] for r in rows) / max(1, sum(r["cite_total"] for r in rows))
    )
    answer_with_gold_cite = sum(1 for r in pos if r["gold_cite"] > 0) / max(1, len(pos))
    refuse_neg = sum(1 for r in neg if r["refused"]) / max(1, len(neg))
    false_refuse = sum(1 for r in pos if r["refused"]) / max(1, len(pos))
    avg_ms = sum(r["ms"] for r in rows) / max(1, len(rows))

    _say("\n" + "=" * 78)
    _say("阶段 1 裸问答原型 · 评测结果")
    _say(f"生成器：{llm.name}　向量：{CONFIG['embed_provider']}　存储：{CONFIG['vector_backend']}"
         f"　拒答阈值：{CONFIG['refuse_threshold']}")
    _say("=" * 78)
    _say(f"{'题号':<6}{'类型':<10}{'Hit@3':<8}{'拒答':<6}{'引用(合法/总)':<16}{'指向gold':<9}{'conf':<7}{'cov':<7}{'备注'}")
    _say("-" * 78)
    for r in rows:
        _say(
            f"{r['id']:<6}{r['type']:<10}{('✓' if r['hit'] else '—'):<8}"
            f"{('是' if r['refused'] else '否'):<6}"
            f"{str(r['cite_valid']) + '/' + str(r['cite_total']):<16}"
            f"{r['gold_cite']:<9}{r['conf']:.3f}  {r['cov']:.2f}   {r['note']}"
        )
    _say("-" * 78)
    _say(f"Hit@3（正样本命中 gold 片段）      ：{hit_at_3:.1%}   （门槛 ≥80%）")
    _say(f"引用准确率（合法引用/全部引用）    ：{cite_precision:.1%}   （门槛 ≥90%）")
    _say(f"回答含 gold 引用比例               ：{answer_with_gold_cite:.1%}")
    _say(f"负样本正确拒答率                   ：{refuse_neg:.1%}   （门槛 100%）")
    _say(f"正样本误拒率                       ：{false_refuse:.1%}")
    _say(f"平均端到端时延                     ：{avg_ms:.0f} ms")
    _say("=" * 78)

    passed = hit_at_3 >= 0.8 and cite_precision >= 0.9 and refuse_neg >= 0.99 and false_refuse <= 0.15
    _say(("结论：达到进入阶段 2 的门槛。" if passed else
          "结论：未达门槛，先修检索或阈值，不要往阶段 2 铺功能。"))
    if llm.name == "mock":
        _say("注意：当前为 mock 抽取式生成器，指标反映的是「检索 + 引用链路」，"
             "真实模型下的忠实度需用 LLM_PROVIDER=openai_compat 重跑。")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# 10. 索引构建
# ---------------------------------------------------------------------------


def cmd_ingest() -> int:
    chunks = load_corpus()
    if not chunks:
        raise SystemExit(f"未在 {KB_DIR} 找到语料")

    embedder = get_embedder()
    _say(f"切片完成：{len(chunks)} 块，正在向量化（{CONFIG['embed_provider']}, dim={CONFIG['embed_dim']}）…")
    vecs = embedder.embed([c.text for c in chunks])
    for c, v in zip(chunks, vecs):
        c.embedding = v

    store = build_store()
    store.rebuild(chunks)

    # 统计：切片分布能直接看出切片策略是否合理
    by_type: Counter = Counter(c.doc_type for c in chunks)
    toks = sorted(c.tokens for c in chunks)
    tables = sum(1 for c in chunks if "|" in c.text[:200])
    _say(f"已建立索引：{INDEX_FILE if CONFIG['vector_backend'] == 'lite' else CONFIG['database_url']}")
    _say(f"  按语料类型：{dict(by_type)}")
    _say(f"  token 分布：min={toks[0]} / P50={toks[len(toks)//2]} / P90={toks[int(len(toks)*0.9)-1]} / max={toks[-1]}")
    _say(f"  含表格的块：{tables} 块（表格整块保留）")
    _say(f"  超过 500 token 的块：{sum(1 for t in toks if t > 500)} 块")
    _say("  提示：本语料单节偏短，多数块低于 300 token 属正常——章节边界优先于填满窗口，")
    _say("        这样引用才能精确指到「§3.2」而不是「§3」。")
    return 0


def cmd_eval() -> int:
    store = build_store()
    store.load()
    return run_eval(Retriever(store, get_embedder()), get_llm())


def cmd_ask(question: str, trace: bool) -> int:
    store = build_store()
    store.load()
    retriever = Retriever(store, get_embedder())
    llm = get_llm()

    out = answer(question, retriever, llm, trace=trace)
    _say("\n" + "-" * 78)
    _say(f"问题：{out['question']}")
    _say("-" * 78)
    _say(out["answer"])
    _say("-" * 78)
    if out["refused"]:
        _say(f"[已拒答] {out['reason']}")
    else:
        c = out["citations"]
        _say(f"[引用] 合法 {len(c['valid'])} 条 / 非法 {len(c['invalid'])} 条")
        llm_res = out["llm"]
        _say(f"[模型] {llm_res.provider}/{llm_res.model}　"
             f"tokens {llm_res.tokens_in}→{llm_res.tokens_out}　"
             f"{llm_res.latency_ms} ms　总耗时 {out['elapsed_ms']} ms")
    _say("[检索依据]")
    for i, (chunk, score, cos, cov, bmn) in enumerate(out["chunks"], 1):
        _say(f"  {i}. {chunk.source_id} §{chunk.section_path} | {ver_label(chunk.version)} | "
             f"score={score:.3f} cos={cos:.3f} cov={cov:.2f} bm25={bmn:.2f}")
    return 0


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="海岳机电 · 阶段 1 裸问答原型（RAG）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="构建索引")
    p_ingest.set_defaults(func=lambda a: cmd_ingest())

    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--trace", action="store_true", help="打印检索过程")
    p_ask.set_defaults(func=lambda a: cmd_ask(a.question, a.trace))

    p_eval = sub.add_parser("eval", help="跑评测集")
    p_eval.set_defaults(func=lambda a: cmd_eval())

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
