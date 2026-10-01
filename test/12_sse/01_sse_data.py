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

    # 为啥是 生成器函数 ?
    # 生成器函数: 没办法一起返回,一部分一部分返回! llm.stream()
    async def generator():
        # 模拟流式结果 1秒 返回一个数字 大概返回10秒
        for index in range(10):
            # yield f"第{index}个结果!" -> 浏览器接收没问题
            # yield f"data: 第{index}个结果!\n\n"
            yield f"data: 第{index}个结果!\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generator(),media_type="text/event-stream")

"""
当前 SSE 协程：
我要等1秒
      ↓
await
      ↓
把控制权交回 Event Loop
      ↓
Event Loop 去处理其他请求
      ↓
1秒后回来继续
"""


@app.get("/html")
def return_html():
    html_path_obj:Path = Path(__file__).parent / "01_sse_data.html"
    return FileResponse(
        path = str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=7777)

"""
Uvicorn
↓
启动服务器

/html
↓
给浏览器 HTML

EventSource("/stream")
↓
建立 SSE 长连接

/stream
↓
StreamingResponse
↓
不断消费 generator()

generator()
↓
yield 一条
↓
await
↓
yield 下一条

浏览器
↓
onmessage
↓
更新页面


拿到 generator()
       ↓
问它：
“有下一条数据吗？”
       ↓
generator yield 一条
       ↓
StreamingResponse 立刻发给浏览器
       ↓
再等下一条
       ↓
再发
       ↓
一直到 generator 结束


generator
= 生产数据

StreamingResponse
= 运输数据

浏览器
= 接收数据


生成器
不断 yield SSE 格式的数据

↓

StreamingResponse
不断发给前端

↓

EventSource
不断接收
"""



