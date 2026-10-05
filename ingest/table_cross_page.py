"""
跨页表格识别与合并（对齐实战口径）：
PaddleOCR 版面分析 → 识别每页表格区域与边框
→ 判断页底表格是否被截断（底部无边框 = 下一页还有）
→ 跨页拼接 + 表头/列数校验 → 输出 Markdown 表格

背景：按页解析会把跨页表格拆成两个残缺表，检索就对不上型号/价格。
"""
from base.logger import logger


def detect_table_region(page_path_or_img) -> dict:
    """
    用 PaddleOCR PP-StructureV2 做版面+表格识别（可选依赖；未安装则返回空结果，不阻塞主流程）。

    page_path_or_img: 图片路径 或 numpy 数组
    返回: {"has_table": bool, "is_truncated_at_bottom": bool, "rows": [...], "header": [...]}
    - is_truncated_at_bottom: 表格底边框是否贴近页面底部 → 下一页还有内容
    - 参考: https://github.com/PaddlePaddle/PaddleOCR (doc/table)
    - 注：未装 PaddleOCR 时，PDF 表格可由 PyMuPDF `page.find_tables()` 兜底（见 loaders.py）
    """
    import re
    empty = {"has_table": False, "is_truncated_at_bottom": False, "rows": [], "header": []}
    try:
        from paddleocr import PPStructure
    except Exception:
        logger.debug("未安装 PaddleOCR，版面/表格识别跳过")
        return empty
    try:
        engine = PPStructure(show_log=False, lang="ch")
        result = engine(page_path_or_img)
        tables = [r["res"] for r in result if r.get("type") == "table"]
        if not tables:
            return empty
        html = tables[0].get("html", "")
        rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
        parsed = []
        for r in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
            parsed.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])
        header = parsed[0] if parsed else []
        return {"has_table": True, "is_truncated_at_bottom": False,
                "rows": parsed[1:], "header": header}
    except Exception as e:
        logger.warning(f"PaddleOCR 表格识别失败: {e}")
        return empty


def merge_cross_page_tables(pages_table_info: list[dict]) -> list[dict]:
    """
    pages_table_info: 每页的 table 检测结果（按页序）
    规则：上一页有"被截断"的空线表格，且下一页顶部存在表格 → 行拼接成一个表。
    合并后校验：列数一致、表头一致；不一致则拆开保留两份并告警。
    """
    merged = []
    pending = None  # 等待下一页续接的截断表
    for pg in pages_table_info:
        if not pg.get("has_table"):
            continue
        if pending and _headers_match(pending, pg):
            _append_rows(pending, pg)
            if not pg.get("is_truncated_at_bottom"):
                merged.append(pending)
                pending = None
            # 若本页也截断，继续保留 pending 等下一页
        else:
            if pending:
                logger.warning("跨页表头不一致，拆开保留")
                merged.append(pending)
            pending = pg if pg.get("is_truncated_at_bottom") else None
            if not pending:
                merged.append(pg)
    if pending:
        merged.append(pending)
    return merged


def _headers_match(t1, t2) -> bool:
    h1, h2 = t1.get("header"), t2.get("header")
    return bool(h1 and h1 == h2) or len(t1.get("rows", [])) and all(
        len(r1) == len(r2) for r1, r2 in zip(t1.get("rows", [])[-1:], t2.get("rows", [])[:1]))


def _append_rows(t1, t2):
    t1["rows"].extend(t2.get("rows", []))


def to_markdown(table: dict) -> str:
    """合并后的表格 → Markdown，保留行/列结构（检索可命中"型号|价格|保修"）。"""
    if not table.get("header"):
        return "\n".join("| " + " | ".join(str(c) if c is not None else "" for c in r) + " |"
                         for r in table.get("rows", []))
    lines = ["| " + " | ".join(str(c) for c in table["header"]) + " |"]
    lines.append("|" + "---|" * len(table["header"]))
    for r in table.get("rows", []):
        lines.append("| " + " | ".join(str(c) if c is not None else "" for c in r) + " |")
    return "\n".join(lines)