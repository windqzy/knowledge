from minio import Minio

from app.infra.config.providers import infra_config
from app.shared.clients import get_minio_client


class MinIOGateway:
    """
    MinIO 对象存储网关类
    统一封装 MinIO 客户端获取、配置读取、图片URL拼接等能力，
    供全项目统一调用，避免到处写配置、重复拼接URL
    """

    @property
    def bucket_name(self) -> str:
        """获取 MinIO 存储桶名称（从全局配置读取）"""
        return infra_config.minio.bucket_name

    @property
    def image_dir(self) -> str:
        """获取 MinIO 中存放图片的目录路径（从全局配置读取）"""
        return infra_config.minio.minio_img_dir

    def client(self) -> Minio:
        """获取 MinIO 客户端实例，用于上传、下载、查询文件等操作"""
        return get_minio_client()

    def build_image_url(self, stem: str, image_name: str) -> str:
        """
        拼接生成 MinIO 图片的可访问URL（HTTP/HTTPS）
        :param stem: 文档名称（不带后缀），用于区分不同文档的图片
        :param image_name: 图片原始文件名
        :return: 可直接访问的 MinIO 图片完整URL
        """
        # 根据配置决定使用 http 还是 https
        protocol = "https" if infra_config.minio.minio_secure else "http"

        # 拼接最终可访问的图片在线地址
        return (
            f"{protocol}://{infra_config.minio.endpoint}/"
            f"{self.bucket_name}{self.image_dir}/{stem}/{image_name}"
        )


# 创建全局唯一的 MinIO 网关实例，全项目复用
minio_gateway = MinIOGateway()