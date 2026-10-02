"""
工具模块，负责提供 SSE 相关的辅助能力。
"""

import asyncio
import json
import threading
from typing import Dict, Any, Optional

from fastapi import Request


class SSEEvent:
    READY = "ready"          # SSE连接建立
    PROGRESS = "progress"    # LangGraph节点进度
    DELTA = "delta"          # LLM流式输出
    FINAL = "final"          # 最终完整答案
    ERROR = "error"          # 错误信息
    CLOSE = "__close__"      # 主动关闭连接


# ============================================================
# 每个 session_id 对应一个 asyncio.Queue
#
# 例如：
#
# {
#     "session_001": asyncio.Queue(),
#     "session_002": asyncio.Queue(),
# }
# ============================================================

_session_stream: Dict[str, asyncio.Queue] = {}


# 每个 Queue 属于哪个 Event Loop
#
# 原因：
# LangGraph 很可能运行在线程池线程中，
# push_to_session() 可能不是 Event Loop 线程调用的。
#
# 所以要找到这个 Queue 所属的 Event Loop，
# 通过 call_soon_threadsafe() 安全地把消息塞进去。
_session_loops: Dict[str, asyncio.AbstractEventLoop] = {}


# 因为 push_to_session 可能从其他线程调用，
# 所以保护上面两个全局字典。
_registry_lock = threading.Lock()


# ============================================================
# 获取 Queue
# ============================================================

def get_sse_queue(session_id: str) -> Optional[asyncio.Queue]:
    """获取指定 session 对应的 asyncio.Queue"""

    with _registry_lock:
        return _session_stream.get(session_id)


# ============================================================
# 创建 Queue
# ============================================================

def create_sse_queue(session_id: str) -> asyncio.Queue:
    """
    为 session 创建 asyncio.Queue。

    注意：
    这个函数应该在 async 接口 / Event Loop 中调用。
    """

    loop = asyncio.get_running_loop()

    with _registry_lock:

        # 已经有 Queue，不重复创建
        old_queue = _session_stream.get(session_id)

        if old_queue is not None:
            return old_queue

        print(f"[SSE] Creating asyncio.Queue for session: {session_id}")

        stream_queue = asyncio.Queue()

        _session_stream[session_id] = stream_queue
        _session_loops[session_id] = loop

        return stream_queue


# ============================================================
# 删除 Queue
# ============================================================

def remove_sse_queue(session_id: str):
    """删除 session 的 Queue 和对应 Event Loop"""

    print(f"[SSE] Removing queue for session: {session_id}")

    with _registry_lock:
        _session_stream.pop(session_id, None)
        _session_loops.pop(session_id, None)


# ============================================================
# SSE协议打包
# ============================================================

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """
    转成标准 SSE 格式。

    最终格式类似：

    event: progress
    data: {"status": "processing"}

    """

    payload = json.dumps(
        data,
        ensure_ascii=False
    )

    return (
        f"event: {event}\n"
        f"data: {payload}\n\n"
    )


# ============================================================
# 生产者：往 Queue 塞消息
# ============================================================

def push_to_session(
    session_id: str,
    event: str,
    data: Dict[str, Any]
) -> bool:
    """
    向指定 session 的 asyncio.Queue 推送消息。

    这个函数故意保持普通 def。

    原因：
    LangGraph 的很多节点本身就是同步 def，
    它们可能在线程池中执行。

    所以：
    工作线程
        ↓
    push_to_session()
        ↓
    loop.call_soon_threadsafe()
        ↓
    Event Loop
        ↓
    queue.put_nowait()
    """

    with _registry_lock:
        stream_queue = _session_stream.get(session_id)
        loop = _session_loops.get(session_id)

    if stream_queue is None or loop is None:
        print(
            f"[SSE] Warning: No queue found for session "
            f"{session_id} when pushing {event}"
        )
        return False

    if loop.is_closed():
        print(
            f"[SSE] Warning: Event loop already closed "
            f"for session {session_id}"
        )
        return False

    message = {
        "event": event,
        "data": data
    }

    try:
        # ----------------------------------------------------
        # 非常关键
        #
        # push_to_session 可能运行在 LangGraph 工作线程中，
        # 不直接跨线程操作 asyncio.Queue。
        #
        # 而是告诉 Queue 所属的 Event Loop：
        #
        # “你自己帮我执行 queue.put_nowait(message)”
        # ----------------------------------------------------

        loop.call_soon_threadsafe(
            stream_queue.put_nowait,
            message
        )

        return True

    except RuntimeError as e:
        print(
            f"[SSE] Failed pushing event {event} "
            f"to session {session_id}: {e}"
        )

        return False


# ============================================================
# 主动关闭某个 SSE
# ============================================================

def close_sse_session(session_id: str):
    """给 Queue 放一个关闭信号"""

    push_to_session(
        session_id,
        SSEEvent.CLOSE,
        {}
    )


# ============================================================
# 消费者：SSE生成器
# ============================================================

async def sse_generator(
    session_id: str,
    request: Request
):
    """
    SSE异步生成器。

    Queue
        ↓
    await queue.get()
        ↓
    event / data
        ↓
    _sse_pack()
        ↓
    yield
        ↓
    StreamingResponse
        ↓
    前端 EventSource
    """

    print(
        f"[SSE] Generator started for session: {session_id}"
    )

    # 获取 Queue
    stream_queue = get_sse_queue(session_id)

    # 如果没有，直接创建
    #
    # 所以无论：
    # /stream 先调用
    # 还是 /query 已经创建
    #
    # 都可以工作。
    if stream_queue is None:
        stream_queue = create_sse_queue(session_id)

    try:

        # ----------------------------------------------------
        # 1. 先告诉前端 SSE 已经建立成功
        # ----------------------------------------------------

        yield _sse_pack(
            SSEEvent.READY,
            {
                "session_id": session_id
            }
        )

        # ----------------------------------------------------
        # 2. 不断等待 Queue 消息
        # ----------------------------------------------------

        while True:

            # 客户端是否主动关闭连接
            if await request.is_disconnected():
                print(
                    f"[SSE] Client disconnected: {session_id}"
                )
                break

            try:

                # ============================================
                # 这里就是改成 asyncio.Queue 后
                # 最核心、最漂亮的地方
                #
                # 老版本：
                #
                # await loop.run_in_executor(
                #     None,
                #     queue.get,
                #     ...
                # )
                #
                # 新版本：
                #
                # await stream_queue.get()
                # ============================================

                msg = await asyncio.wait_for(
                    stream_queue.get(),
                    timeout=1.0
                )

            except asyncio.TimeoutError:

                # 1秒没消息很正常
                #
                # 回到 while 顶部继续判断：
                # 客户端是不是已经断开
                continue

            event = msg.get("event")
            data = msg.get("data", {})

            # ------------------------------------------------
            # 收到关闭事件
            # ------------------------------------------------

            if event == SSEEvent.CLOSE:
                print(
                    f"[SSE] Closing signal received: {session_id}"
                )
                break

            # ------------------------------------------------
            # 转成SSE协议并发送
            # ------------------------------------------------

            yield _sse_pack(
                event,
                data
            )

    except asyncio.CancelledError:

        # 浏览器关闭页面 / EventSource关闭
        print(
            f"[SSE] Generator cancelled: {session_id}"
        )

        return

    except (
        ConnectionResetError,
        BrokenPipeError
    ):

        print(
            f"[SSE] Connection closed: {session_id}"
        )

        return

    except Exception as e:

        print(
            f"[SSE] Exception for session "
            f"{session_id}: {e}"
        )

        # 尝试给前端发错误事件
        try:
            yield _sse_pack(
                SSEEvent.ERROR,
                {
                    "message": str(e)
                }
            )
        except Exception:
            pass

    finally:

        print(
            f"[SSE] Generator finished: {session_id}"
        )

        remove_sse_queue(session_id)
"""
工具模块，负责提供 SSE 相关的辅助能力。
"""

import asyncio
import json
import threading
from typing import Dict, Any, Optional

from fastapi import Request


class SSEEvent:
    READY = "ready"          # SSE连接建立
    PROGRESS = "progress"    # LangGraph节点进度
    DELTA = "delta"          # LLM流式输出
    FINAL = "final"          # 最终完整答案
    ERROR = "error"          # 错误信息
    CLOSE = "__close__"      # 主动关闭连接


# ============================================================
# 每个 session_id 对应一个 asyncio.Queue
#
# 例如：
#
# {
#     "session_001": asyncio.Queue(),
#     "session_002": asyncio.Queue(),
# }
# ============================================================

_session_stream: Dict[str, asyncio.Queue] = {}


# 每个 Queue 属于哪个 Event Loop
#
# 原因：
# LangGraph 很可能运行在线程池线程中，
# push_to_session() 可能不是 Event Loop 线程调用的。
#
# 所以要找到这个 Queue 所属的 Event Loop，
# 通过 call_soon_threadsafe() 安全地把消息塞进去。
_session_loops: Dict[str, asyncio.AbstractEventLoop] = {}


# 因为 push_to_session 可能从其他线程调用，
# 所以保护上面两个全局字典。
_registry_lock = threading.Lock()


# ============================================================
# 获取 Queue
# ============================================================

def get_sse_queue(session_id: str) -> Optional[asyncio.Queue]:
    """获取指定 session 对应的 asyncio.Queue"""

    with _registry_lock:
        return _session_stream.get(session_id)


# ============================================================
# 创建 Queue
# ============================================================

def create_sse_queue(session_id: str) -> asyncio.Queue:
    """
    为 session 创建 asyncio.Queue。

    注意：
    这个函数应该在 async 接口 / Event Loop 中调用。
    """

    loop = asyncio.get_running_loop()

    with _registry_lock:

        # 已经有 Queue，不重复创建
        old_queue = _session_stream.get(session_id)

        if old_queue is not None:
            return old_queue

        print(f"[SSE] Creating asyncio.Queue for session: {session_id}")

        stream_queue = asyncio.Queue()

        _session_stream[session_id] = stream_queue
        _session_loops[session_id] = loop

        return stream_queue


# ============================================================
# 删除 Queue
# ============================================================

def remove_sse_queue(session_id: str):
    """删除 session 的 Queue 和对应 Event Loop"""

    print(f"[SSE] Removing queue for session: {session_id}")

    with _registry_lock:
        _session_stream.pop(session_id, None)
        _session_loops.pop(session_id, None)


# ============================================================
# SSE协议打包
# ============================================================

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """
    转成标准 SSE 格式。

    最终格式类似：

    event: progress
    data: {"status": "processing"}

    """

    payload = json.dumps(
        data,
        ensure_ascii=False
    )

    return (
        f"event: {event}\n"
        f"data: {payload}\n\n"
    )


# ============================================================
# 生产者：往 Queue 塞消息
# ============================================================

def push_to_session(
    session_id: str,
    event: str,
    data: Dict[str, Any]
) -> bool:
    """
    向指定 session 的 asyncio.Queue 推送消息。

    这个函数故意保持普通 def。

    原因：
    LangGraph 的很多节点本身就是同步 def，
    它们可能在线程池中执行。

    所以：
    工作线程
        ↓
    push_to_session()
        ↓
    loop.call_soon_threadsafe()
        ↓
    Event Loop
        ↓
    queue.put_nowait()
    """

    with _registry_lock:
        stream_queue = _session_stream.get(session_id)
        loop = _session_loops.get(session_id)

    if stream_queue is None or loop is None:
        print(
            f"[SSE] Warning: No queue found for session "
            f"{session_id} when pushing {event}"
        )
        return False

    if loop.is_closed():
        print(
            f"[SSE] Warning: Event loop already closed "
            f"for session {session_id}"
        )
        return False

    message = {
        "event": event,
        "data": data
    }

    try:
        # ----------------------------------------------------
        # 非常关键
        #
        # push_to_session 可能运行在 LangGraph 工作线程中，
        # 不直接跨线程操作 asyncio.Queue。
        #
        # 而是告诉 Queue 所属的 Event Loop：
        #
        # “你自己帮我执行 queue.put_nowait(message)”
        # ----------------------------------------------------

        loop.call_soon_threadsafe(
            stream_queue.put_nowait,
            message
        )

        return True

    except RuntimeError as e:
        print(
            f"[SSE] Failed pushing event {event} "
            f"to session {session_id}: {e}"
        )

        return False


# ============================================================
# 主动关闭某个 SSE
# ============================================================

def close_sse_session(session_id: str):
    """给 Queue 放一个关闭信号"""

    push_to_session(
        session_id,
        SSEEvent.CLOSE,
        {}
    )


# ============================================================
# 消费者：SSE生成器
# ============================================================

async def sse_generator(
    session_id: str,
    request: Request
):
    """
    SSE异步生成器。

    Queue
        ↓
    await queue.get()
        ↓
    event / data
        ↓
    _sse_pack()
        ↓
    yield
        ↓
    StreamingResponse
        ↓
    前端 EventSource
    """

    print(
        f"[SSE] Generator started for session: {session_id}"
    )

    # 获取 Queue
    stream_queue = get_sse_queue(session_id)

    # 如果没有，直接创建
    #
    # 所以无论：
    # /stream 先调用
    # 还是 /query 已经创建
    #
    # 都可以工作。
    if stream_queue is None:
        stream_queue = create_sse_queue(session_id)

    try:

        # ----------------------------------------------------
        # 1. 先告诉前端 SSE 已经建立成功
        # ----------------------------------------------------

        yield _sse_pack(
            SSEEvent.READY,
            {
                "session_id": session_id
            }
        )

        # ----------------------------------------------------
        # 2. 不断等待 Queue 消息
        # ----------------------------------------------------

        while True:

            # 客户端是否主动关闭连接
            if await request.is_disconnected():
                print(
                    f"[SSE] Client disconnected: {session_id}"
                )
                print('--------------------断开连接-----------------------------------')
                break

            try:

                # ============================================
                # 这里就是改成 asyncio.Queue 后
                # 最核心、最漂亮的地方
                #
                # 老版本：
                #
                # await loop.run_in_executor(
                #     None,
                #     queue.get,
                #     ...
                # )
                #
                # 新版本：
                #
                # await stream_queue.get()
                # ============================================

                msg = await asyncio.wait_for(
                    stream_queue.get(),
                    timeout=1.0
                )

            except asyncio.TimeoutError:

                # 1秒没消息很正常
                #
                # 回到 while 顶部继续判断：
                # 客户端是不是已经断开
                continue

            event = msg.get("event")
            data = msg.get("data", {})

            # ------------------------------------------------
            # 收到关闭事件
            # ------------------------------------------------

            if event == SSEEvent.CLOSE:
                print(
                    f"[SSE] Closing signal received: {session_id}"
                )
                break

            # ------------------------------------------------
            # 转成SSE协议并发送
            # ------------------------------------------------

            yield _sse_pack(
                event,
                data
            )

    except asyncio.CancelledError:

        # 浏览器关闭页面 / EventSource关闭
        print(
            f"[SSE] Generator cancelled: {session_id}"
        )

        return

    except (
        ConnectionResetError,
        BrokenPipeError
    ):

        print(
            f"[SSE] Connection closed: {session_id}"
        )

        return

    except Exception as e:

        print(
            f"[SSE] Exception for session "
            f"{session_id}: {e}"
        )

        # 尝试给前端发错误事件
        try:
            yield _sse_pack(
                SSEEvent.ERROR,
                {
                    "message": str(e)
                }
            )
        except Exception:
            pass

    finally:

        print(
            f"[SSE] Generator finished: {session_id}"
        )

        remove_sse_queue(session_id)