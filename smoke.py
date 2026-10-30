"""
端到端冒烟测试：不用打开浏览器就能验证全部接口。

用法：
    # 先起服务
    python -m uvicorn app.main:app --port 8020
    # 再跑
    python scripts/smoke.py [http://127.0.0.1:8020]

它验证的是「契约」，不是「答案好不好」：
  事件顺序、错误码、会话归属、多轮上下文、引用落库、原文回查。
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8020").rstrip("/")
API = BASE + "/api/v1"

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [通过] {name}")
    else:
        failed += 1
        print(f"  [失败] {name}  {detail}")


# 环境里可能配了全局代理，本机回环必须绕过，否则连接会被代理重置
client = httpx.Client(trust_env=False, timeout=60)


def login(username: str, password: str = "123456") -> dict:
    r = client.post(f"{API}/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['data']['access_token']}"}


def chat(headers: dict, question: str, session_id: str | None = None, debug: bool = False):
    """返回 (事件列表, 拼好的正文)。"""
    events: list[tuple[str, dict]] = []
    with client.stream(
        "POST", f"{API}/chat?debug={1 if debug else 0}", headers=headers,
        json={"session_id": session_id, "question": question},
    ) as resp:
        if resp.status_code != 200:
            body = resp.read().decode()
            return [], body
        ev = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line[5:].strip())))
    text = "".join(d["text"] for e, d in events if e == "delta")
    return events, text


def main() -> int:
    print(f"\n目标：{API}\n")

    print("1. 健康检查")
    h = client.get(f"{API}/health").json()
    check("知识库就绪", h["data"]["kb_ready"] is True, str(h))
    print(f"       chunks={h['data'].get('chunks')} provider={h['data'].get('provider')}")

    print("\n2. 鉴权")
    H = login("t-001")
    bad = client.post(f"{API}/auth/login", json={"username": "t-001", "password": "wrong"})
    check("错误密码返回 40101", bad.json()["code"] == 40101, bad.text)
    noauth = client.post(f"{API}/chat", json={"question": "x"})
    check("无 token 返回 40101", noauth.json()["code"] == 40101, noauth.text)

    print("\n3. 首轮提问（新建会话）")
    ev, text = chat(H, "更换 M-0912 主轴轴承时，前端轴承压盖的预紧力矩是多少？", debug=True)
    names = [e for e, _ in ev]
    meta = next(d for e, d in ev if e == "meta")
    done = next(d for e, d in ev if e == "done")
    session_id = meta["session_id"]
    check("新建会话由服务端返回", bool(meta["new_session"]) and session_id.startswith("sess_"))
    check("事件顺序 meta 开头、done 结尾", names[0] == "meta" and names[-1] == "done", str(names[:3]))
    check("有 delta 流式片段", names.count("delta") > 10, f"delta={names.count('delta')}")
    check("有 citation 事件", "citations" in names)
    check("回答未拒答", done["refused"] is False)
    check("回答含正确力矩", "45" in text, text[:80])
    check("回答带引用标记", "[来源:" in text or "[来源：" in text, text[:80])

    print("\n4. 第二轮（同一会话，验证上下文）")
    ev2, text2 = chat(H, "那后端轴承呢？", session_id=session_id)
    meta2 = next(d for e, d in ev2 if e == "meta")
    done2 = next(d for e, d in ev2 if e == "done")
    check("沿用同一会话", meta2["session_id"] == session_id and meta2["new_session"] is False)
    check("指代问题仍能作答（未拒答）", done2["refused"] is False, text2[:80])
    check("命中后端力矩 25", "25" in text2, text2[:120])

    print("\n5. 会话与历史")
    sessions = client.get(f"{API}/sessions", headers=H).json()["data"]["items"]
    check("会话出现在列表", any(s["id"] == session_id for s in sessions))
    msgs = client.get(f"{API}/sessions/{session_id}/messages", headers=H).json()["data"]["items"]
    check("历史 4 条消息", len(msgs) == 4, f"实际 {len(msgs)}")
    bot = [m for m in msgs if m["role"] == "assistant"]
    check("助手消息带引用", len(bot) == 2 and all(len(m["citations"]) > 0 for m in bot))
    check("引用存了版本号", all(c["version"] for m in bot for c in m["citations"]))
    first_cite = bot[0]["citations"][0]
    check("引用 rank 从 1 开始", first_cite["index"] == 1)

    print("\n6. 引用原文回查")
    detail = client.get(f"{API}/citations/{first_cite['chunk_id']}", headers=H).json()
    check("能取到原文", detail["code"] == 0 and "45" in detail["data"]["text"], str(detail)[:120])
    check("带章节与版本", bool(detail["data"]["section_path"]) and detail["data"]["version"] == "V3.2")

    print("\n7. 拒答（正常业务结果）")
    ev3, text3 = chat(H, "员工出差的住宿报销标准是多少？")
    done3 = next(d for e, d in ev3 if e == "done")
    check("拒答标记为 true", done3["refused"] is True)
    check("拒答走 done 不走 error", "error" not in [e for e, _ in ev3])
    check("拒答文案正确", "转人工" in text3, text3)

    print("\n8. 越权与会话隔离")
    A = login("admin")
    stolen = client.post(f"{API}/chat", headers=A,
                         json={"session_id": session_id, "question": "随便问问"}).json()
    check("访问他人会话返回 40301", stolen["code"] == 40301, str(stolen))

    print("\n9. 入参与错误码")
    cases = [
        ("空问题", {"question": "   "}, 40001),
        ("超长问题", {"question": "x" * 501}, 40002),
        ("不存在的会话", {"session_id": "sess_nope", "question": "安全措施"}, 40401),
    ]
    for name, body, code in cases:
        r = client.post(f"{API}/chat", headers=H, json=body)
        check(f"{name} 返回 {code}", r.json()["code"] == code, r.text[:120])

    print("\n10. 数据域（f41）")
    # t-001 在 AREA-A，点名 AREA-B 的 H-1023 应被拦；管理员不受限
    blocked = client.post(f"{API}/chat", headers=H,
                          json={"question": "H-1023 液压夹具夹不紧怎么处理？"}).json()
    check("跨区设备返回 40302", blocked["code"] == 40302, str(blocked)[:120])
    check("提示里带设备号与所属区", "H-1023" in blocked["message"] and "A 区" in blocked["message"],
          blocked.get("message", ""))
    allowed = client.post(f"{API}/chat?debug=0", headers=A,
                          json={"question": "H-1023 液压夹具夹不紧怎么处理？"})
    check("管理员可跨区提问", allowed.status_code == 200, str(allowed.status_code))
    own = client.post(f"{API}/chat", headers=H,
                      json={"question": "M-0912 主轴异响怎么判断？"})
    check("本区设备不被拦", own.status_code == 200, str(own.status_code))

    print("\n11. 会话关闭（软删）")
    d = client.delete(f"{API}/sessions/{session_id}", headers=H).json()
    check("关闭成功", d["code"] == 0)
    after = client.get(f"{API}/sessions", headers=H).json()["data"]["items"]
    check("列表不再出现", all(s["id"] != session_id for s in after))

    print(f"\n{'=' * 46}\n通过 {passed} 项，失败 {failed} 项\n{'=' * 46}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
