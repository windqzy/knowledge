import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import  FileResponse , StreamingResponse
from pydantic import BaseModel

#fastapi定义路径:  1. 创建fastapi对象 app = FastApi()  2. @app.函数(/路径 -> 具体路径 | /路径/{}) 函数
#fastapi定义请求方式: 2. @app. get / post / put / delete  (/路径 -> 具体路径 | /路径/{}) 函数  [选择哪个? 需不需要请求体! 动作!]
#fastapi如何接收参数: 1. 路径参数  2. param/query ?key=value 3. 请求体参数[json/文件]
"""
  参数接收: 
     路径参数: 1. 设置动态路径 {key} {key}  2. 形参列表中定义对应名称的参数即可  key:类型 <- 路径参数必须传递 不能添加=默认值
     param参: 1. 定义 ? key同名的参数  2. 指定对应的类型,如果不是必须传递 给与默认 = None | 值 3. param不影响路径设计
     请求体参数接收: 
             json参数: 1. 定义对应的类型 继承BaseModel 属性等于jsonKey [对象 == 对象]
             文件参数: 1. 定义UploadFile类型接受 2. read() .file读取文件内容 3. 完成数据转存
  响应数据:
     响应json字符串: dict | 继承BaseModel类型 [推荐]
     响应一个文件 : 返回一个文件
     响应流式数据[查询铺垫]
  测试fastapi:  /docs -> 测试页面 sw...
"""

# 1.删除指定id的商品!
# id -> 唯一标识 | 也不是隐私数据 -> path传递参数  /product/{id}
# 删除 - delete
# 模拟删除逻辑: id = x 被删除成功了 ..
# 给前端一段删除结果 {code:0 失败 1 成功} -> json  dict / BaseModel类

app = FastAPI()

@app.delete("/product/{id}")
def delete_product(id:int):
    # 调用我们的逻辑
    print(f"已经删除id={id}的商品信息!!")
    return {
        "code":"1",
        "msg":"已经删除对应的商品!"
    }

# 2. 查询价格大于100 名字中包含'电脑'的商品信息
# 请求方式 get
# 传参方式 param ? price = 100 & keyword = "电脑"
# 路径  /product/find
@app.get("/product/find")
def find_product(price:float,keyword:str|None = None):
    print(f"接收参数:{price},{keyword}")
    return {
        "code": "1",
        "msg": "已经查询对应的商品!",
        "data":["娃哈哈","可口可乐","乐事薯片","飞天老窖"]
    }

# 3. 保存商品信息 { id name price location }  json
# 请求方式 post
# 路径设计 product/save
# 传参方式 json  {"":"",}

# python接收json数据 定义个类型,继承BaseModel
# json本身在前端就是一个对象表达形式 JSON  JavaScriptObjectNotation {} "json字符串"  == python后端 应该使用对象接收
# 继承BaseModel普通的类型没有json转化能力  [json字符串 <=> 对象]
class Location(BaseModel):
    p:str
    c:str
    z:str

class Product(BaseModel):
    id:int
    name:str
    price:float
    location:Location

# 类 -> @dataclass  1. 类的简化创建和使用 __init__  [普通工具类 配置类]
# 类 -> TypedDict   使用字典的形式进行对象的操作  Langgraph state  return {}
# 类 -> BaseModel   1. 类的简化创建和使用 __init__  2. 严格模式 类型校验  3. json转化函数  [fastapi json]
@app.post("/product/save")
def save_product(product:list[Product]):
    print(f"接收到参数:{product}")
    return {
        "code":1,
        "msg": "商品数据保存成功!",
        "data": product
    }


# 4. 前端传递了一张商品图片! 保存起来
# 请求: put / post
# 地址: product/upload
# 参数: 文件  image=图片

from app.shared.runtime.logger import PROJECT_ROOT

@app.post("/product/upload")
async def upload_image(image:UploadFile):
    """

       image =  UploadFile -> 组合数据  包含上传文件 也包含上传的文件信息 文件名 类型 大小..
       属性: filename -> 文件名 / size 字节大小 / content_type mimetype类型  / file 文件的引用流对象 open(wb) as file  file.read()
       函数: async image.read() -> 直接读取字节数据 == .file.read()
    :param image:
    :return:
    """
    # 接收文件存储到output中
    # 定义个output/文件名 -> Path
    image_path_obj:Path = PROJECT_ROOT / "output" / image.filename
    data = await image.read()
    image_path_obj.write_bytes(data)
    return {
        "msg":f"{image.filename}图片保存成功!"
    }


# 5. 前端传递一个图片的名字,我们从output中找到,并且返回
# 前端请求方式 get image/find?name=xx
# param | path
# 返回一个文件
from mimetypes import guess_type

@app.get("/image/find")
def find_image(name:str='逻辑'):


    image_path_obj: Path = PROJECT_ROOT / "output" / f"{name}.jpg"
    # 我们想返回文件 -> 文件装到响应体 -> 二进制
    # 我们想返回文件 -> 文件地址 ->  FileResponse(path) -> 读取文件 -> 放到响应体 -> 前端
    #return json -> {key:"xx"}
    #return json -> {key:"xx"} base64
    return FileResponse(
        filename=image_path_obj.name,  # 文件名 带有后缀名   响应头: content_disposition = attachment;filename=文件名.xx  [下载头 附件头]
        path = str(image_path_obj),    # 文件地址 -> 读取 -> 响应体
        media_type =guess_type(image_path_obj.name)[0]  # mimetype
    )

# 6. 响应流式数据
async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流

@app.get("/stream")
async def stream_response():
    # StreamingResponse 返回类型
    # 参数1: 生成器 多久生成一个数据,我就可以随时返回!   <-     大语言模型.stream("chunk")
    # 参数2: media_type="text/event-stream"
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn  # 服务器 tomcat apache nginx...
    uvicorn.run(app=app,host="0.0.0.0",port=8888)