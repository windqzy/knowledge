from pymilvus import MilvusClient

from app.infra.config.providers import infra_config
from app.shared.clients import get_milvus_client


class MilvusGateway:
    @property
    def chunk_collection_name(self):
        return infra_config.milvus.chunks_collection

    @property
    def item_name_collection_name(self):
        return infra_config.milvus.item_name_collection


    #返回客户端
    @property
    def milvus_client(self):
        return get_milvus_client()

milvus_gateway = MilvusGateway()