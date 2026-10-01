from pydantic import BaseModel


# 接口status
class StatusResponseSchema(BaseModel):
    code: int = 200
    task_id: str
    status: str
    done_list: list[str]
    running_list: list[str]


class UploadResponseSchema(BaseModel):
    code: int
    message: str
    task_ids: list[str]
