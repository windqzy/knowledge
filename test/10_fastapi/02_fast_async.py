import asyncio
import threading
import time

from fastapi import FastAPI, BackgroundTasks

# 理解: fastapi所有接口本身就是互相独立的! 不干扰!
# 例如:    @app.get(product/{id})   第一次调用  独立协程 / 独立线程
#                                  第二次调用  独立协程 / 独立线程       [宏观]  async 函数 -> 协程 | 没有加 -> 线程
#                                  第三次调用  独立协程 / 独立线程

# 异步技术?? [微观]  一个函数内  第一次调用 [ 1 2 [开启异步] 3 [不等第二步执行完] ] 线程 协程
# 方式1: asyncio.create_task(函数()) -> 加入  |  asyncio.run(函数()) [报错] 入口 开启循环事件 +
# 方式2: 使用线程方式
# 方式3: fastapi 函数(task:BackgroundTasks)  task.add_task(函数())
#       1. 开启线程或者协程运行目标函数  [ 1 2[独立线程和协程 10s] 3[不阻塞] ]
#       2. 真实内容 2 执行的顺序在3之后!  不阻塞 快速返回 异步执行

# fastapi线程工作模式:
#       协程事件的线程 31800 -> 跑所有的协程函数 async    time.sleep(10) 不行  asyncio.sleep(10)
#      工作线程(线程池)[x,x,x,]  -> 没有async - 新的线程 复用原有线程   time.sleep(10)

app = FastAPI()

def invoke(name:str,age:int):
    thread_id = threading.get_ident()
    print(f"invoke 运行的线程id:{thread_id}")  # 33272
    print("2")  # 耗时动作
    time.sleep(5)
    print(f"异步任务执行完毕")

@app.get("/product/1")
def product1(task:BackgroundTasks):
    print("1")
    thread_id = threading.get_ident() #当前代码到底在哪个线程执行
    print(f"/product/1 运行的线程id:{thread_id}")  #33272
    # 2
    task.add_task(invoke,name="二狗子",age=18)  # 1. 异步 线程/协程 函数  2. 执行循序 宏观 1 2[异步] 3  微观  1 3 [2]
    time.sleep(5)
    print("3")

    return {
        "msg":"执行成功"
    }


@app.get("/product/2")
async def product2():
    thread_id = threading.get_ident()
    print(f"/product/2 运行的线程id:{thread_id}")
    # time.sleep(5)
    await asyncio.sleep(5)
    return '完成'


if __name__ == "__main__":
    import uvicorn  # 服务器 tomcat apache nginx...
    uvicorn.run(app=app,host="0.0.0.0",port=9999)


"""
FastAPI收到很多请求
        ↓
 ┌──────────────┬──────────────┐
 ↓              ↓
async def       def
 ↓              ↓
协程            线程池
 ↓              ↓
event loop      工作线程

async def → 协程，适合等待网络/数据库；普通 def → FastAPI 通常放到线程池执行。

"""