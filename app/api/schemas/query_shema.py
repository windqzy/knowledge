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

class ClearHistoryResponseSchema(BaseModel):
    message:str
    deleted_count:int

class SearchHistoryItemResponseSchema(BaseModel):
    id:str
    session_id:str
    role:str #user assistant
    text:str
    rewritten_query:str
    item_names:list[str]
    image_urls:list[str]
    ts:float

class SearchHistoryResultResponseSchema(BaseModel):
    session_id:str
    items:list[SearchHistoryItemResponseSchema]

