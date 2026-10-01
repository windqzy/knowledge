from typing import Any

from pymilvus import DataType, MilvusClient

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log, logger


# 1. 获取参数并校验
@step_log("get_data_and_validate")
def get_data_and_validate(state: ImportGraphState) -> list[dict[str, Any]]:
    # chunks
    chunks = state.get("embeddings_content")
    # 非空校验
    if not chunks:
        logger.error(f"chunks为空,业务无法继续,提前终止!!")
        raise ValueError(f"chunks为空,业务无法继续,提前终止!!")
    # 返回数据
    return chunks


# 2.准备chunk对应的向库
step_log('prepare_item_name_milvus_collection')


def prepare_chunks_milvus_collection():
    # 1. 加载milvus的客户端对象
    milvus_client = milvus_gateway.milvus_client

    # 判断是否有这个集合
    has_item_name_collection = milvus_client.has_collection(collection_name=milvus_gateway.chunk_collection_name)
    if has_item_name_collection:
        # 如果 kb_item_names 已经存在，不重新创建，直接加载它，然后继续使用。
        milvus_client.load_collection(collection_name=milvus_gateway.chunk_collection_name)
        logger.info(f"{milvus_gateway.chunk_collection_name}已经存在无需创建!!")
        return

    # 2.创建集合对应的schema[字段类型]
    # {chunk_id,file_title,item_name,parent_title,title,part,content,dense_vector,sparse_vector}
    milvus_schema = milvus_client.create_schema(
        auto_id=True,  # 主键自增长
        enable_dynamic_field=True  # 允许额外字段
    )
    milvus_schema.add_field(field_name='chunk_id', datatype=DataType.INT64, is_primary=True)
    milvus_schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name='parent_title', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=512, nullable=True)
    milvus_schema.add_field(field_name='part', datatype=DataType.INT64)
    milvus_schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=65535, nullable=True)
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

    index_params.add_index(
        field_name="item_name",
        index_type="INVERTED",
        index_name="item_name_index"
    )
    # 4. 创建集合数据
    milvus_client.create_collection(
        collection_name=milvus_gateway.chunk_collection_name,
        schema=milvus_schema,
        index_params=index_params
    )
    # 5. 加载load集合,否则不能使用
    milvus_client.load_collection(collection_name=milvus_gateway.chunk_collection_name)


# 3. 删除并插入数据 [二次插入/文档可能更新/file_title]
@step_log("insert_chunks_to_milvus")
def insert_chunks_to_milvus(chunks: list[dict[str, Any]]) -> list:
    milvus_client: MilvusClient = milvus_gateway.milvus_client
    # 1. 先根据file_title删除数据
    file_title = chunks[0].get("file_title")
    milvus_client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title == '{file_title}'"
    )
    # 2. 插入新的数据
    result = milvus_client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=chunks
    )
    # 3. 获取回显id返回
    logger.info(f"元数据条数:{len(chunks)},本次插入:{result.get('insert_count')}")
    # 4. 主键回显思路
    if len(chunks) == len(result.get('ids')):
        # 正常回显
        ids: list = result.get("ids", [])
        for chunk, id in zip(chunks, ids):
            chunk['chunk_id'] = id
    else:
        # 插入的并不是都成功
        logger.warning(f"元数据条数:{len(chunks)},本次插入:{result.get('insert_count')},两者数量不等,请检查处理!")


def index_chunks(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取参数并校验
    chunks = get_data_and_validate(state)
    # 2.准备chunk对应的向库
    prepare_chunks_milvus_collection()
    # 3. 删除并插入数据 [二次插入/文档可能更新/file_title]
    insert_chunks_to_milvus(chunks)
    # 4.更新并回显数据
    state['chunks'] = chunks
    return state
