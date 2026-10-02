from pydantic import BaseModel

class QueryRequestSchema(BaseModel):
    query:str
    session_id:str
    is_stream:bool

class AsyncQueryResponseSchema(BaseModel):
    message:str
    session_id:str

class SyncQueryResponseSchema(BaseModel):
    message:str
    session_id:str
    answer:str
    done_list:list[str]
    image_urls:list[str]
