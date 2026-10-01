import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse , FileResponse
from pathlib import Path
from mimetypes import guess_type



app = FastAPI()
# 需求1: 后端单纯使用流式响应,返回多段数据!!
# 接口 流式响应的接口
@app.get("/stream")
async def return_stream():

    # 返回两种事件类型! 1. 下载的进度事件 带有下载的进度 [processing]  2. 下载完成事件 下载完成 [completed]

    async def generator():
        # 模拟流式结果 1秒 返回一个数字 大概返回10秒
        for index in range(101):
            yield f"event: processing\n"
            yield f"data: {index}\n\n"
            await asyncio.sleep(0.1)
        yield f"event: completed\n"
        yield f"data: <font color='#fb5832'>下载完成</font>\n\n"

    return StreamingResponse(generator(),media_type="text/event-stream")
"""
虽然是两个 yield，但是前端收到后会把它们当成连续的数据流：
event: processing\n
data: 10\n\n

拼起来就是：
event: processing
data: 10


注意最后这个：
\n\n

它才是在告诉浏览器：
这一整个 SSE 事件结束了，可以触发 JS 回调了。

第一个 yield
↓
event: processing\n
↓
浏览器：
“收到 event 字段了，但事件还没结束，先等等”

第二个 yield
↓
data: 10\n\n
↓
浏览器看到 \n\n
↓
“好了，这个事件完整了”

于是得到：

事件类型 = processing
数据 = 10
↓
触发：

addEventListener("processing", ...)
"""
@app.get("/html")
def return_html():
    html_path_obj:Path = Path(__file__).parent / "02_sse_data.html"
    return FileResponse(
        path = str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8888)



"""
你现在：
yield "event: processing\n"
yield f"data: {index}\n\n"


也可以直接写成：
yield f"event: processing\ndata: {index}\n\n"


我更推荐第二种，因为一眼就能看出来：
这是一个完整 SSE 事件
"""

