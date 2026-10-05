"""配置管理：读 config.ini + 环境变量（环境变量优先）"""
import configparser
import os


class Config:
    def __init__(self, config_file=None):
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'data', 'ecommerce')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'models')  # 放 bge-m3 / bge-reranker / bert 分类器

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file, encoding='utf-8')

        # MySQL
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='127.0.0.1'))
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.getint('mysql', 'port', fallback=3307)))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='ecommerce_qa'))

        # Redis
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='127.0.0.1'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.getint('redis', 'port', fallback=6379)))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='123456'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='127.0.0.1'))
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self.config.getint('milvus', 'port', fallback=19530)))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME', self.config.get('milvus', 'database_name', fallback='ecom'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', self.config.get('milvus', 'collection_name', fallback='ecom_rag'))

        # LLM
        self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='qwen-plus'))
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback=''))
        self.DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL', self.config.get('llm', 'dashscope_base_url',
                                                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'))

        # 检索参数
        self.PARENT_CHUNK_SIZE = int(os.getenv('PARENT_CHUNK_SIZE', self.config.getint('retrieval', 'parent_chunk_size', fallback=1000)))
        self.CHILD_CHUNK_SIZE = int(os.getenv('CHILD_CHUNK_SIZE', self.config.getint('retrieval', 'child_chunk_size', fallback=200)))
        self.PARENT_CHUNK_OVERLAP = int(os.getenv('PARENT_CHUNK_OVERLAP', self.config.getint('retrieval', 'parent_chunk_overlap', fallback=150)))
        self.CHILD_CHUNK_OVERLAP = int(os.getenv('CHILD_CHUNK_OVERLAP', self.config.getint('retrieval', 'child_chunk_overlap', fallback=30)))
        self.RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', self.config.getint('retrieval', 'retrieval_k', fallback=10)))
        self.CANDIDATE_M = int(os.getenv('CANDIDATE_M', self.config.getint('retrieval', 'candidate_m', fallback=2)))
        self.BM25_THRESHOLD = float(os.getenv('BM25_THRESHOLD', self.config.getfloat('retrieval', 'bm25_threshold', fallback=0.85)))

        # 模型设备（auto → 有 CUDA 用 GPU，否则 CPU）
        self.MODEL_DEVICE = os.getenv('MODEL_DEVICE', self.config.get('model', 'device', fallback='auto'))

        # 应用
        self.VALID_SOURCES = eval(os.getenv('VALID_SOURCES', self.config.get('app', 'valid_sources', fallback='["手机","电脑"]')))
        self.CUSTOMER_SERVICE_PHONE = os.getenv('CUSTOMER_SERVICE_PHONE', self.config.get('app', 'customer_service_phone', fallback='400-000-0000'))

        self.LOG_FILE = os.path.join(self.PROJECT_ROOT, 'logs', 'app.log')


config = Config()

if __name__ == '__main__':
    print(config.MYSQL_PORT, config.RETRIEVAL_K, config.VALID_SOURCES)