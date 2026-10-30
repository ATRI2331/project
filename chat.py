"""聊天主接口（SSE）与会话管理。

SSE 事件顺序：meta → status(检索中) → retrieval → status(生成中) → delta ×N
             → citations → done；故障走 error，拒答走 done(refused=true)。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import rag
from ..config import DEBUG_SCORES, HISTORY_TOKEN_BUDGET, HISTORY_TURNS, QUESTION_MAX_LEN, SSE_PING_SECONDS
from ..deps import check_rate_limit, current_user, request_id
from ..errors import AppError, ErrCode, envelope
from ..rag import stream_answer, suggest_title
from ..scope import describe_block, out_of_scope_model
from ..store import (
    add_citations,
    add_message,
    close_conversation,
    create_conversation,
    get_chunk_detail,
    get_conversation,
    list_conversations,
    list_messages,
    recent_context,
    touch_conversation,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])

# 「它/这个/那」这类指代，单看本轮问题检索必挂，要和上一轮问题拼起来
_ANAPHORA = ("它", "这个", "那个", "上面", "刚才", "该", "其")


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str


def _sse(event: str, data: dict[str, Any]) -> str:
    """SSE 帧。ensure_ascii=False 是为了中文在浏览器 Network 面板里直接可读。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _with_heartbeat(source: AsyncIterator[str], ping_seconds: float) -> AsyncIterator[str]:
    """
    给事件流套一层心跳：ping_seconds 内没有新事件就发一行注释帧 `: ping`。
    少了这层，Nginx 60 秒无数据会直接断流，用户看到的就是「回答打一半停了」。
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except Exception as exc:  # 交给下游统一转成 error 事件
            await queue.put(exc)
        finally:
            await queue.put(sentinel)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=ping_seconds)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        task.cancel()


def _retrieval_question(question: str, history: list[dict[str, str]]) -> str:
    last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
    if not last_user:
        return question
    if len(question) <= 8 or any(k in question for k in _ANAPHORA):
        return f"{last_user} {question}"
    return question


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user: dict[str, Any] = Depends(current_user),
    debug: int = Query(default=0, ge=0, le=1, description="1=下发相似度分数，调试用"),
) -> Response:
    """
    注意：所有校验都在「开始推流之前」完成，失败时返回标准 JSON 信封。
    一旦开始推流，HTTP 状态码就固定为 200 了，错误只能走 SSE 的 error 事件。
    """
    rid = request_id(request)
    question = (body.question or "").strip()
    if not question:
        raise AppError(ErrCode.BAD_QUESTION)
    if len(question) > QUESTION_MAX_LEN:
        raise AppError(ErrCode.QUESTION_TOO_LONG,
                       f"问题超出长度上限（{QUESTION_MAX_LEN} 字），请精简后重试")
    if not rag.kb_ready():
        raise AppError(ErrCode.KB_NOT_READY)

    # 数据域（f41）：问题点名了别的作业区的设备就拦。放在推流之前，
    # 这样前端拿到的是标准 JSON 信封而不是一条 error 事件，便于直接弹提示。
    blocked = out_of_scope_model(question, user)
    if blocked:
        raise AppError(ErrCode.FORBIDDEN_SCOPE, describe_block(blocked, user))

    check_rate_limit(user["id"])

    # 会话归属：别人的会话要明确拒绝，不能当成「不存在」，否则排查时看不出是越权
    if body.session_id:
        conv = get_conversation(body.session_id)
        if not conv or conv["status"] != "active":
            raise AppError(ErrCode.SESSION_NOT_FOUND)
        if conv["user_id"] != user["id"]:
            raise AppError(ErrCode.FORBIDDEN_SESSION)
        session = conv
        is_new = False
    else:
        session = create_conversation(user["id"], suggest_title(question))
        is_new = True

    # 历史必须在写入本轮提问之前取，否则会把当前问题当成上下文重复一次
    history = recent_context(session["id"], HISTORY_TURNS, HISTORY_TOKEN_BUDGET)
    retrieval_q = _retrieval_question(question, history)

    add_message(session["id"], "user", question, request_id=rid)
    touch_conversation(session["id"], turn_delta=1)

    show_scores = bool(debug and DEBUG_SCORES)

    async def event_gen() -> AsyncIterator[str]:
        t0 = time.time()
        yield _sse("meta", {
            "session_id": session["id"],
            "new_session": is_new,
            "request_id": rid,
        })
        yield _sse("status", {"stage": "retrieving", "text": "正在检索手册与案例…"})

        answer_text = ""
        refused = False
        items: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        error_code = 0
        started_generating = False

        try:
            async for ev in stream_answer(retrieval_q, history):
                kind = ev["type"]

                if kind == "retrieval":
                    if not started_generating:
                        started_generating = True
                        yield _sse("status", {"stage": "generating", "text": "正在生成回答…"})
                    if show_scores:
                        yield _sse("debug", {
                            "confidence": ev["confidence"],
                            "coverage": ev["coverage"],
                            "top": ev["top"],
                        })
                elif kind == "delta":
                    answer_text += ev["text"]
                    yield _sse("delta", {"text": ev["text"]})
                elif kind == "retract":
                    # 已推出去的正文没有有效引用，要求前端整段替换为拒答文案
                    answer_text = ev["text"]
                    refused = True
                    yield _sse("retract", {"text": ev["text"], "reason": ev["reason"]})
                elif kind == "citations":
                    items = ev["items"]
                    payload = items if show_scores else [
                        {k: v for k, v in it.items() if k != "score"} for it in items
                    ]
                    yield _sse("citations", {"items": payload})
                elif kind == "refused":
                    refused = True
                elif kind == "done":
                    answer_text = ev["text"]
                    refused = bool(ev["refused"])
                    usage = {"tokens_in": ev.get("tokens_in", 0),
                             "tokens_out": ev.get("tokens_out", 0),
                             "model": ev.get("model", "")}
        except Exception as exc:
            # 流内故障：保留已推送内容，前端提示中断，并把这个错误也落库便于复盘
            error_code = int(ErrCode.LLM_FAILED)
            yield _sse("error", {"code": error_code,
                                 "message": "模型服务超时，请重试或转人工",
                                 "detail": str(exc)[:200]})

        elapsed = int((time.time() - t0) * 1000)
        mid = add_message(
            session["id"], "assistant", answer_text, refused=refused, request_id=rid,
            tokens_in=usage.get("tokens_in", 0), tokens_out=usage.get("tokens_out", 0),
            model=usage.get("model", ""), latency_ms=elapsed, error_code=error_code,
        )
        if items:
            add_citations(mid, items)
        touch_conversation(session["id"])

        yield _sse("done", {
            "session_id": session["id"],
            "message_id": mid,
            "refused": refused,
            "elapsed_ms": elapsed,
            "usage": usage,
        })

    return StreamingResponse(
        _with_heartbeat(event_gen(), SSE_PING_SECONDS),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 不加这一条，Nginx 会缓冲整个响应，流式直接失效
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def sessions(
    request: Request,
    user: dict[str, Any] = Depends(current_user),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return envelope(data={"items": list_conversations(user["id"], limit)},
                    request_id=request_id(request))


@router.post("/sessions")
async def new_session(request: Request, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    conv = create_conversation(user["id"], "新会话")
    return envelope(data=conv, request_id=request_id(request))


@router.get("/sessions/{session_id}/messages")
async def session_messages(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    conv = get_conversation(session_id)
    if not conv:
        raise AppError(ErrCode.SESSION_NOT_FOUND)
    if conv["user_id"] != user["id"]:
        raise AppError(ErrCode.FORBIDDEN_SESSION)
    return envelope(data={"session": {"id": conv["id"], "title": conv["title"]},
                          "items": list_messages(session_id)},
                    request_id=request_id(request))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    conv = get_conversation(session_id)
    if not conv:
        raise AppError(ErrCode.SESSION_NOT_FOUND)
    if conv["user_id"] != user["id"]:
        raise AppError(ErrCode.FORBIDDEN_SESSION)
    close_conversation(session_id)   # 软删，审计要留痕
    return envelope(data={"closed": session_id}, request_id=request_id(request))


# ---------------------------------------------------------------------------
# 引用原文
# ---------------------------------------------------------------------------

@router.get("/citations/{chunk_id}")
async def citation_detail(
    chunk_id: str,
    request: Request,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    detail = get_chunk_detail(chunk_id)
    if not detail:
        raise AppError(ErrCode.SESSION_NOT_FOUND, "引用原文不存在，可能已被替换")
    return envelope(data=detail, request_id=request_id(request))
