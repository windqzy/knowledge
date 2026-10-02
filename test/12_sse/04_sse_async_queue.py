import time

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app.shared.utils.sse_utils import (
    SSEEvent,
    close_sse_session,
    create_sse_queue,
    get_sse_queue,
    push_to_session,
    sse_generator,
)

app = FastAPI()


# ============================================================
# 1. 模拟 LangGraph + LLM
# ============================================================
def mock_long_task(session_id: str):
    """
    模拟：
    LangGraph 节点进度
        ↓
    LLM 流式输出
        ↓
    最终完成
    """

    # --------------------------------
    # 模拟 LangGraph 5 个节点
    # --------------------------------
    for i in range(1, 6):

        push_to_session(
            session_id,
            SSEEvent.PROGRESS,
            {
                "status": "processing",
                "step": i,
                "message": f"正在执行第 {i} 步",
            },
        )

        time.sleep(1)

    # --------------------------------
    # 模拟 LLM 流式输出
    # --------------------------------
    answer = "你好，这是一段模拟的 LLM 流式回答。"

    final_text = ""

    for ch in answer:

        final_text += ch

        push_to_session(
            session_id,
            SSEEvent.DELTA,
            {
                "delta": ch
            },
        )

        time.sleep(0.08)

    # --------------------------------
    # 最终完整结果
    # --------------------------------
    push_to_session(
        session_id,
        SSEEvent.FINAL,
        {
            "status": "completed",
            "answer": final_text,
            "image_urls": [],
        },
    )

    # 稍等一下
    time.sleep(0.5)

    # 通知 SSE 关闭
    close_sse_session(session_id)


# ============================================================
# 2. 启动任务接口
# ============================================================
@app.get("/start/{session_id}")
async def start_task(
    session_id: str,
    background_tasks: BackgroundTasks
):

    print(f"[TEST] start task: {session_id}")

    # --------------------------------
    # 非常重要：
    # 在 Event Loop 中先创建 asyncio.Queue
    # --------------------------------
    if get_sse_queue(session_id) is None:

        create_sse_queue(session_id)

    # --------------------------------
    # mock_long_task 是同步 def
    #
    # BackgroundTasks 会让它在工作线程执行
    # --------------------------------
    background_tasks.add_task(
        mock_long_task,
        session_id
    )

    return {
        "message": "测试任务已启动",
        "session_id": session_id,
    }


# ============================================================
# 3. SSE 流式接口
# ============================================================
@app.get("/stream/{session_id}")
async def stream(
    session_id: str,
    request: Request
):

    print(f"[TEST] stream connected: {session_id}")

    # 如果没有 Queue
    # 就创建一个
    if get_sse_queue(session_id) is None:

        create_sse_queue(session_id)

    return StreamingResponse(

        sse_generator(
            session_id=session_id,
            request=request
        ),

        media_type="text/event-stream",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 4. 测试页面
# ============================================================
@app.get("/")
async def index():

    # 注意这里使用 r"""
    #
    # r = raw string
    #
    # 防止 Python 把 JavaScript 中的
    # \n 提前解析成真正换行
    return HTMLResponse(
        r"""
<!DOCTYPE html>

<html lang="zh-CN">

<head>

    <meta charset="UTF-8">

    <title>SSE asyncio.Queue 测试</title>

</head>


<body>

    <h2>SSE asyncio.Queue 测试</h2>


    <button onclick="startTest()">
        开始测试
    </button>


    <p>
        状态：
        <span id="status">
            未开始
        </span>
    </p>


    <p>
        当前进度：
        <span id="progress">
            0
        </span>
    </p>


    <p>
        LLM 流式回答：
    </p>


    <div
        id="answer"
        style="
            white-space: pre-wrap;
            border: 1px solid #ccc;
            padding: 10px;
            min-height: 50px;
        "
    >
    </div>


    <hr>


    <p>
        SSE 事件日志：
    </p>


    <pre id="log"></pre>



<script>

let es = null;


/*
========================================
日志显示
========================================
*/
function log(text) {

    const box =
        document.getElementById("log");

    box.textContent +=
        text + "\n";
}


/*
========================================
点击开始测试
========================================
*/
async function startTest() {

    // 每次测试生成新的 session_id
    const sessionId =
        "test-" + Date.now();


    console.log(
        "sessionId:",
        sessionId
    );


    // 清空页面
    document.getElementById(
        "status"
    ).innerText = "启动中";


    document.getElementById(
        "progress"
    ).innerText = "0";


    document.getElementById(
        "answer"
    ).innerText = "";


    document.getElementById(
        "log"
    ).innerText = "";


    /*
    ========================================
    第一步：
    启动后台任务

    后端：
        创建 asyncio.Queue
        ↓
        BackgroundTasks
        ↓
        mock_long_task
    ========================================
    */

    const response =
        await fetch(
            "/start/" + sessionId
        );


    const result =
        await response.json();


    log(
        "START: " +
        JSON.stringify(result)
    );


    /*
    ========================================
    第二步：
    建立 SSE 长连接
    ========================================
    */

    es =
        new EventSource(
            "/stream/" + sessionId
        );


    /*
    ========================================
    READY
    ========================================
    */

    es.addEventListener(
        "ready",
        function(event) {

            console.log(
                "READY",
                event.data
            );

            log(
                "READY: " +
                event.data
            );

            document.getElementById(
                "status"
            ).innerText =
                "SSE 已连接";
        }
    );


    /*
    ========================================
    PROGRESS
    LangGraph 节点进度
    ========================================
    */

    es.addEventListener(
        "progress",
        function(event) {

            const data =
                JSON.parse(
                    event.data
                );


            console.log(
                "PROGRESS",
                data
            );


            log(
                "PROGRESS: " +
                event.data
            );


            document.getElementById(
                "status"
            ).innerText =
                data.message;


            document.getElementById(
                "progress"
            ).innerText =
                data.step + " / 5";
        }
    );


    /*
    ========================================
    DELTA
    LLM 一个字一个字输出
    ========================================
    */

    es.addEventListener(
        "delta",
        function(event) {

            const data =
                JSON.parse(
                    event.data
                );


            console.log(
                "DELTA",
                data
            );


            log(
                "DELTA: " +
                event.data
            );


            document.getElementById(
                "answer"
            ).innerText +=
                data.delta;
        }
    );


    /*
    ========================================
    FINAL
    最终完整答案
    ========================================
    */

    es.addEventListener(
        "final",
        function(event) {

            const data =
                JSON.parse(
                    event.data
                );


            console.log(
                "FINAL",
                data
            );


            log(
                "FINAL: " +
                event.data
            );


            document.getElementById(
                "status"
            ).innerText =
                "完成";


            document.getElementById(
                "answer"
            ).innerText =
                data.answer;
        }
    );


    /*
    ========================================
    SSE 断开
    ========================================
    */

    es.onerror =
        function(event) {

            console.log(
                "SSE CLOSED",
                event
            );


            log(
                "SSE 连接关闭"
            );


            if (es) {

                es.close();

                es = null;
            }
        };
}

</script>


</body>

</html>
        """
    )


# ============================================================
# 5. 启动
# ============================================================
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8899
    )