"""报销接口的响应组装：抽取置信度、导出材料与导出时间戳。"""

from .schemas import ExpenseExportMaterial
from .storage import ExpenseMaterial


def extraction_confidence(material: ExpenseMaterial) -> float:
    """金额/日期/商户三个抽取字段的命中占比（0~1）。

    旧实现 `1.0 if material.summary else 0.0` 恒为 1.0（summary 永不为空），
    是假指标；现在反映真实抽取质量，也是复盘指标"字段纠正次数"的分母参照。
    """

    signals = (
        material.extracted_amount is not None,
        bool(material.extracted_date),
        bool(material.merchant),
    )
    return round(sum(signals) / len(signals), 2)


def export_material(material: ExpenseMaterial) -> ExpenseExportMaterial:
    """导出清单里的材料条目（不含 raw_text 等大字段）。"""

    return ExpenseExportMaterial(
        material_id=material.material_id,
        claim_id=material.claim_id,
        title=material.title,
        doc_type=material.doc_type,
        extracted_amount=material.extracted_amount,
        extracted_date=material.extracted_date,
        merchant=material.merchant,
        file_uri=material.file_uri,
    )

