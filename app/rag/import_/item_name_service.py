from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, DataType

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway


# 1. 获取并校验参数 (chunks/file_title)
@step_log('get_data_and_validate')
def get_data_and_validate(state) -> tuple[list, str]:
    # 1.获取参数
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    # 2.非空校验
    if not chunks:
        logger.error(f'chunks内容为空，后续业务无法继续，提前终止！！')
        raise ValueError(f'chunks内容为空，后续业务无法继续，提前终止！！')
    if not file_title:
        file_title = 'defaultSS'
    return chunks, file_title


# 2. 调用模型识别item_name
@step_log('call_model_recognition_item_name')
def call_model_recognition_item_name(chunks, file_title) -> str:
    # 2.1 加载大语言模型客户端
    llm_model = llm_provider.chat()
    # 2.2 拼接context内容,通过chunks拼接 -> top5
    context = ''
    for chunk in chunks[:5]:
        context += f'标题：{chunk.get('file_title')},内容：{chunk.get('content')}\n'
    # 2.3 加载提示词文件,并且动态替换关键字(file_title/context)
    recognition_prompt: str = load_prompt('item_name_recognition', file_title=file_title, context=context)
    """
    请从以下信息中识别出商品名称与型号：
    文件名：{file_title}
    
    正文切片（用于辅助识别）：
    {context}
    
    要求：
    1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
    2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
    3. 如果无法识别商品名称,请返回空字符串。
    """

    # 2.4 准备HumanMessage | Chains
    messages = [HumanMessage(
        content=recognition_prompt
    )]

    chains = llm_model | StrOutputParser()
    # 2.5 调用模型进行item_name识别.并获取返回结果
    item_name = chains.invoke(messages)
    if not item_name:
        logger.info(f"item_name没有识别到,使用file_title:{file_title}赋予默认值!!")
        item_name = file_title
    # 2.6 返回即可
    return item_name


# 5. 提前准备item_name对应的集合
@step_log('prepare_item_name_milvus_collection')
def prepare_item_name_milvus_collection():
    # 1. 加载milvus的客户端对象
    milvus_client = milvus_gateway.milvus_client

    # 判断是否有这个集合
    has_item_name_collection = milvus_client.has_collection(collection_name=milvus_gateway.item_name_collection_name)
    if has_item_name_collection:
        # 如果 kb_item_names 已经存在，不重新创建，直接加载它，然后继续使用。
        milvus_client.load_collection(collection_name=milvus_gateway.item_name_collection_name)
        logger.info(f"{milvus_gateway.item_name_collection_name}已经存在无需创建!!")
        return

    # 2.创建集合对应的schema[字段类型]
    milvus_schema = milvus_client.create_schema(
        auto_id=True,  # 主键自增长
        enable_dynamic_field=True  # 允许额外字段
    )
    milvus_schema.add_field(field_name='pk', datatype=DataType.INT64, is_primary=True)
    milvus_schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    milvus_schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 3. 根据需求创建索引 [提高查询效率]
    """
     索引: 加快查询速度
     怎么加快: 索引是一个额外的高效数据类型 
        没有索引之前 -> 数据(全表查询) -> On
        有索引了之后 -> 索引 (hash/红黑树) -> ologn -> 数据
     创建索引的前提: 频繁查询的字段

     1. 项目的流程 图结构
     2. 每个点的作用和操作数据
     3. 节点的内部宏观步骤
     4. 节点设计的技术点
    """
    index_params = milvus_client.prepare_index_params()
    # 稠密向量
    index_params.add_index(
        field_name="dense_vector",  # 给哪个field添加索引 列名
        index_name="dense_index",
        index_type="HNSW",  # 索引类型  思路1: AUTOINDEX 字段的类型自动选择  思路2: 自己指定
        metric_type="COSINE",  # 向量相似度比较类型  稠密向量 IP  / COSINE / L2
        params={
            "M": 64,  # 每个点最大的邻居节点数量
            "efConstruction": 100  # 从100个点选出的M的值
        }  # Index building params
    )

    """
    浮点矩阵 -> 稠密
    FLAT: 全盘搜索,性能最差,召回率最高!! FLAT索引是最简单、最直接的浮点向量索引和搜索方法之一。它依赖于一种 "蛮力 "方法，即直接将每个查询向量与数据集中的每个向量进行比较，而无需任何高级预处理或数据结构。这种方法保证了准确性，由于对每个潜在匹配都进行了评估，因此可提供 100% 的召回率。
    HNSW: 效率比IVF慢,但是召回率IVF高, 数据分成若干图层,上层的稀疏 -> 下层密集 (类似地图)
    IVF_FLAT: 效率快,召回率稍差 分区,每个区有中心点,先根据中心点选中一个区域,在区域内使用FLAT检索  (nlist:划分数据集的簇数。)
    """
    # 稀疏向量
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",  # SPARSE_INVERTED_INDEX 02_倒排索引
        metric_type="IP",  # IP  / BM25 被淘汰了
        params={"inverted_index_algo": "DAAT_MAXSCORE"}  # 优化跳过小值! 直接计算大值区间
    )
    # 4. 创建集合数据
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_name_collection_name,
        schema=milvus_schema,
        index_params=index_params
    )
    # 5. 加载load集合,否则不能使用
    milvus_client.load_collection(collection_name=milvus_gateway.item_name_collection_name)


# 6. 保存item_name信息到对应的集合中
@step_log('insert_into_milvus_collection')
def insert_into_milvus_collection(item_name, file_title):
    # 1.item_name 生成稠密和稀糊向量
    result = llm_provider.generate_embeddings([item_name])
    dense_vector = result['dense'][0]
    sparse_vector = result['sparse'][0]
    # 2.file_title作为条件 先删除对应数据
    milvus_client: MilvusClient = milvus_gateway.milvus_client
    milvus_client.delete(collection_name=milvus_gateway.item_name_collection_name,
                         filter=f"file_title == '{file_title}'")
    insert_dict: dict = milvus_client.insert(collection_name=milvus_gateway.item_name_collection_name,
                                             data={
                                                 'file_title': file_title,
                                                 'item_name': item_name,
                                                 'dense_vector': dense_vector,
                                                 'sparse_vector': sparse_vector
                                             })
    logger.info(
        f"完成{item_name}数据的查询,插入的条数:{insert_dict.get('insert_count')},对应的主键:{insert_dict.get('ids')}")


@step_log('recognize_and_index_item_name')
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取并校验参数 (chunks/file_title)
    chunks, file_title = get_data_and_validate(state)
    # 2. 调用模型识别item_name
    item_name = call_model_recognition_item_name(chunks, file_title)
    # 3. 更新chunks的内容 [{没有 item_name <- item_name }]
    for chunk in chunks:
        chunk['item_name'] = item_name
    # 4. 更新state chunks | item_name
    state['item_name'] = item_name
    state['chunks'] = chunks
    # 5. 提前准备item_name对应的集合
    prepare_item_name_milvus_collection()
    # 6. 保存item_name信息到对应的集合中
    insert_into_milvus_collection(item_name, file_title)
    # 7. 返回state return state
    return state


"""
state
│
├─ chunks
└─ file_title
      ↓
① get_data_and_validate()
校验数据
      ↓
② call_model_recognition_item_name()
让 LLM 看前5个 Chunk
      ↓
item_name = "HAK 180 烫金机"
      ↓
③ 给每个 Chunk 加 item_name
      ↓
④ 更新 state
      ↓
⑤ prepare_item_name_milvus_collection()
准备 Milvus 表 + 索引
      ↓
⑥ insert_into_milvus_collection()
item_name → Dense + Sparse
      ↓
保存到 Milvus
      ↓
return state


              向量索引

FLAT
= 全部搜
= 准但慢

IVF
= 先分区，再搜部分区域
= 更快

IVF_FLAT
= IVF分区 + 区内FLAT

IVF_SQ8
= IVF分区 + SQ压缩
= 更省内存

IVF_PQ
= IVF分区 + PQ强压缩
= 更省内存，精度损失更大

HNSW
= 建图 + 沿邻居找
= 快 + 召回高 + 比较吃内存

AUTOINDEX
= 让Milvus帮你选

SPARSE_INVERTED_INDEX
= 专门给Sparse稀疏向量

                  一个 Chunk
                      ↓
                   BGE-M3
                /             \
               ↓               ↓
        Dense Vector      Sparse Vector
        [1024个数字]      {ID:权重}
               ↓               ↓
             HNSW         02_倒排索引
               ↓               ↓
        快速找附近向量     快速找共同特征
               ↓               ↓
           COSINE              IP
               ↓               ↓
          语义相似分数      特征匹配分数
                \             /
                 \           /
                  ↓         ↓
                  最终混合检索
"""
