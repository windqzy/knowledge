# ====================== 全局配置 ======================
# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
# 混合检索的limit的数量 5
HYBRID_SEARCH_LIMIT_NUMBER = 5
HYBRID_SEARCH_RANKER_WEIGHTS =(0.5, 0.5)

RERANK_MAX_TOPK: int = 10  # 动态截取数量,最多10个!
RERANK_MIN_TOPK: int = 3   # 动态截取数量,最少3个!
RERANK_GAP_RATIO: float = 0.2 # 断崖的百分比
RERANK_GAP_ABS: float = 0.2   # 断崖的分数绝对值
RERANK_MAX_INPUT_TOKENS: int = 508 # 上下文的有效窗口数量 514 - 2 - 4
RERANK_SUMMARY_CHAR_RATIO: float = 1.3 # 算出来 answer压缩后是多少token [1300token]! -> 中文字符和token的比 1.3->   大语言模型 字符 1000
RERANK_MIN_SUMMARY_CHARS: int = 50  # 算 10个字符 -> 大模型 -> 最少50,保证压缩的有效性!!!  [ 截取 压缩总结不完善 ]


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
