import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import HYBRID_SEARCH_LIMIT_NUMBER
from app.shared.runtime.logger import step_log, logger


# 1. 获取和校验参数
@step_log("get_data_and_validate")
def get_data_and_validate(state: QueryGraphState) -> str:
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.info(f"rewritten_query为空,业务无法继续,提前终止!!")
        raise ValueError(f"rewritten_query为空,业务无法继续,提前终止!!")
    return rewritten_query


# 2. 使用重写问题进行联网搜索
@step_log("web_search_bailian_mcp")
async def web_search_bailian_mcp(rewritten_query: str):
    # openai的mcp服务
    # https://openai.github.io/openai-agents-python/mcp/#2-streamable-http-mcp-servers
    # 1.连接mcp服务
    async with MCPServerStreamableHttp(
            name="bailian web search",
            params={
                "url": infra_config.mcp.mcp_base_url,
                "headers": {"Authorization": f"Bearer {infra_config.mcp.api_key}"},
                "timeout": 30,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
    ) as server:
        # 2. 获取工具列表
        tool_list = await server.list_tools()
        logger.debug(f"查询到当前服务下的工具列表:{tool_list}")
        # 3. 调用工具列表
        result = await server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": HYBRID_SEARCH_LIMIT_NUMBER
            }
        )
        logger.debug(f"调用bailian_web_search工具,搜索:{rewritten_query},对应结果:{result}")
        return result


@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> QueryGraphState:
    # 1.获取并校验参数
    rewritten_query: str = get_data_and_validate(state)
    # 2.调用mcp服务查询网络得到焦距
    result = asyncio.run(web_search_bailian_mcp(rewritten_query))
    logger.info(f'结果：{result}')
    result_json: str = result.content[0].text
    # print(result_json)
    result_dict: dict = json.loads(result_json)
    web_search_docs_mcp:list[dict] = result_dict.get('pages',[])
    web_search_docs:list[dict] = [
        {
            'content':item.get('snippet',''),
            'title':item.get('title',''),
            'url':item.get('url',''),
            'type':'web',
        }
        for item in web_search_docs_mcp
    ]
    logger.info(f"基于:{rewritten_query}问题,进行联网搜索,获取的答案为:{web_search_docs}")
    state['web_search_docs'] = web_search_docs
    return state

    """
    {"pages":[
        {"snippet":"公告和警告 HP Color LaserJet Pro M180-M181 多功能打印机系列 
        选择类别 所有类别 标题 严重程度 类别 更新时间 无警报  *您的产品可能与所示图片略有不同。  
        HP Color LaserJet Pro M180-M181 多功能打印机系列  产品背面 电池下面 在条形码上 
        若为笔记本电脑,请按 Fn + Esc 若为台式电脑,请按 Ctrl + Alt + s 若为Chromebooks,
        请在登录屏幕上按 Alt + v 对于Poly 产品,在主机、单独的控制元件或充电底座上。 
        在此处查找特定产品的位置。",
        "hostname":"无",
        "hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico",
        "title":"https://www.support.hp.com/cn-zh/product/details/hp-color-laserjet-pro-m180-m181-multifunction-printer-series/14135030",
        "url":"https://www.support.hp.com/cn-zh/product/details/hp-color-laserjet-pro-m180-m181-multifunction-printer-series/14135030"},
        {"snippet":"国家/地区中华人民共和国","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico",
        "title":"Settings | 中国惠普","url":"https://www.hp.com/cn-zh/settings.html"},
        {"snippet":"Sorry! This page is broken. Unlike our products. It might have moved, or the URL could be incorrect. You can always return to our Homepage. .","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"https://www.hp.com/ca-en/shop/product.aspx?id=6N4E8AA&opt=ABA&sel=MTO","url":"https://www.hp.com/ca-en/shop/product.aspx?id=6N4E8AA&opt=ABA&sel=MTO"},{"snippet":"SAVE S$100 S$199.00 S$66.33 * FREE* 2 + 3-month Instant Ink trial  (0) Print at home like a Pro. Fax included.  S$299.00 SAVE S$100 (33%)  S$199.00 Installment from  S$66.33 * A4 Color Inkjet All-in-One Printers, Perfect for Business Ink Printers Functions: Print, copy, scan, fax Print Speed (Black): Up to 29 ppm Apple AirPrint™; Ethernet networking; USB; Wireless (Wi-Fi®); Wireless direct printing; Fax HP Instant Ink eligible; Automatic document feeder; Touchscreen; Quiet mode; Print over VPN with HP+ Overview  1 2 3 4 5 *For selected products only Reliable technology uniquely built to work at home Be more productive at home with quiet mode, HP's most reliable Wi-Fi®, and HP Wolf Essential Security.* Print, scan, and copy from the comfort of your couch with the best and easiest-to-use print app.* Less hassle, more printing Always ready to print Join 12 million+ people enjoying HP's first smart ink delivery service and never run out of ink**. Save up to 50% with Instant Ink for peace of mind and printing flexibility. Choose printers with 45% recycled plastic and help protect our forests with each print**. Recycle your Original HP ink cartridges for free with the HP Planet Partners program.*","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"HP OfficeJet Pro 8130e All-in-One Printer - Includes 3 Months of FREE printing with Instant Ink","url":"https://www.hp.com/sg-en/shop/printers/business-printers/officejet/hp-officejet-pro-8130e-all-in-one-printer-40q48b.html"},{"snippet":"YN系列热转印机在塑胶产品表面Logo图案转印参数设置  2026-07-24  在塑胶产品表面进行Logo或图案的热转印,参数设置是决定最终效果的关键。东莞市优耐机械科技有限公司针对塑胶产品,参数设置主要围绕温度、压力、速度和时间四个核心要素。温度是首要参数,通常塑胶产品的转印温度在120-180℃之间,具体取决于塑胶的材质(如ABS、PP、PE等)和热转印膜的耐温特性。东莞市优耐机械科技有限公司生产的YN系列热转印机,凭借其精密的控制系统和稳定的机械结构,能够帮助用户轻松实现高质量的转印效果。","hostname":"","hostlogo":"","title":"真空热转印机-全自动烫金机-热转印设备生产厂家-东莞市优耐机械科技有限公司","url":"https://dgyounaijx.com/col.jsp?id=104"}],"request_id":"eae2837e-e461-9598-8ee4-6592a6748df1","tools":[],"status":0}2026-10-06 10:39:19.556 | INFO     | web_search_service.py:search_by_web  :57   - 结果：meta=None content=[TextContent(type='text', text='{"pages":[{"snippet":"公告和警告 HP Color LaserJet Pro M180-M181 多功能打印机系列 选择类别 所有类别 标题 严重程度 类别 更新时间 无警报  *您的产品可能与所示图片略有不同。  HP Color LaserJet Pro M180-M181 多功能打印机系列  产品背面 电池下面 在条形码上 若为笔记本电脑,请按 Fn + Esc 若为台式电脑,请按 Ctrl + Alt + s 若为Chromebooks,请在登录屏幕上按 Alt + v 对于Poly 产品,在主机、单独的控制元件或充电底座上。 在此处查找特定产品的位置。","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"https://www.support.hp.com/cn-zh/product/details/hp-color-laserjet-pro-m180-m181-multifunction-printer-series/14135030","url":"https://www.support.hp.com/cn-zh/product/details/hp-color-laserjet-pro-m180-m181-multifunction-printer-series/14135030"},{"snippet":"国家/地区中华人民共和国","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"Settings | 中国惠普","url":"https://www.hp.com/cn-zh/settings.html"},{"snippet":"Sorry! This page is broken. Unlike our products. It might have moved, or the URL could be incorrect. You can always return to our Homepage. .","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"https://www.hp.com/ca-en/shop/product.aspx?id=6N4E8AA&opt=ABA&sel=MTO","url":"https://www.hp.com/ca-en/shop/product.aspx?id=6N4E8AA&opt=ABA&sel=MTO"},{"snippet":"SAVE S$100 S$199.00 S$66.33 * FREE* 2 + 3-month Instant Ink trial  (0) Print at home like a Pro. Fax included.  S$299.00 SAVE S$100 (33%)  S$199.00 Installment from  S$66.33 * A4 Color Inkjet All-in-One Printers, Perfect for Business Ink Printers Functions: Print, copy, scan, fax Print Speed (Black): Up to 29 ppm Apple AirPrint™; Ethernet networking; USB; Wireless (Wi-Fi®); Wireless direct printing; Fax HP Instant Ink eligible; Automatic document feeder; Touchscreen; Quiet mode; Print over VPN with HP+ Overview  1 2 3 4 5 *For selected products only Reliable technology uniquely built to work at home Be more productive at home with quiet mode, HP\'s most reliable Wi-Fi®, and HP Wolf Essential Security.* Print, scan, and copy from the comfort of your couch with the best and easiest-to-use print app.* Less hassle, more printing Always ready to print Join 12 million+ people enjoying HP\'s first smart ink delivery service and never run out of ink**. Save up to 50% with Instant Ink for peace of mind and printing flexibility. Choose printers with 45% recycled plastic and help protect our forests with each print**. Recycle your Original HP ink cartridges for free with the HP Planet Partners program.*","hostname":"无","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01cuA6Dw1WCvM1zP1YK_!!6000000002753-73-tps-16-16.ico","title":"HP OfficeJet Pro 8130e All-in-One Printer - Includes 3 Months of FREE printing with Instant Ink","url":"https://www.hp.com/sg-en/shop/printers/business-printers/officejet/hp-officejet-pro-8130e-all-in-one-printer-40q48b.html"},{"snippet":"YN系列热转印机在塑胶产品表面Logo图案转印参数设置  2026-07-24  在塑胶产品表面进行Logo或图案的热转印,参数设置是决定最终效果的关键。东莞市优耐机械科技有限公司针对塑胶产品,参数设置主要围绕温度、压力、速度和时间四个核心要素。温度是首要参数,通常塑胶产品的转印温度在120-180℃之间,具体取决于塑胶的材质(如ABS、PP、PE等)和热转印膜的耐温特性。东莞市优耐机械科技有限公司生产的YN系列热转印机,凭借其精密的控制系统和稳定的机械结构,能够帮助用户轻松实现高质量的转印效果。","hostname":"","hostlogo":"","title":"真空热转印机-全自动烫金机-热转印设备生产厂家-东莞市优耐机械科技有限公司","url":"https://dgyounaijx.com/col.jsp?id=104"}],"request_id":"eae2837e-e461-9598-8ee4-6592a6748df1","tools":[],"status":0}', annotations=None, meta=None)] structuredContent=None isError=False


    """
    return state
