# 电商客服 RAG · 应用镜像
# 构建：docker build -t ecom-rag:1.0 .
# 说明：模型(models/)与数据(data/)不进镜像，运行时用 volume 挂载（见 .dockerignore / docker-compose.app.yml）

FROM python:3.10.20-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 系统依赖（编译 + 健康检查）
RUN apt-get update && apt-get install -y \
    gcc g++ curl git zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 依赖（利用层缓存：先只拷贝 requirements）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade "pip<24.1" && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 若容器为纯 CPU、想省掉 CUDA 体积，可改用下面这行单独装 CPU 版 torch：
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 项目代码（models/data/logs/venv 等已被 .dockerignore 排除）
COPY . .

EXPOSE 8003

# 健康检查（服务与依赖都就绪才算健康）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
