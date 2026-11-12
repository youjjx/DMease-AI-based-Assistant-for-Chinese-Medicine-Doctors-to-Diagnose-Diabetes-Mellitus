"""UTF-8 exports and a self-contained printable analysis report."""
from __future__ import annotations

import csv
import html
import io
import json

SOFTWARE_NAME = "DMease中医糖尿病智能辅助诊疗系统"
VERSION = "V1.0"


def json_export(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def csv_export(rows: list[dict]) -> bytes:
    if not rows:
        return b"\xef\xbb\xbf"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return stream.getvalue().encode("utf-8-sig")


def report_html(payload: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    patient = payload["patient"]
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"
    sections = [f"<h1>{SOFTWARE_NAME}</h1><p>{VERSION} 辅助分析报告</p>",
                f"<p>患者标识：{esc(patient['name'])}　生成时间：{esc(payload['created_at'])}</p>",
                f"<p>年龄：{esc(patient['age'])}　性别：{esc(patient['sex'])}　体质：{esc(patient['constitution'])}</p>",
                f"<h2>主诉与输入</h2><p>{esc(patient['complaint'])}</p>",
                f"<p>过敏：{esc('、'.join(patient['allergies']) or '未填写')}；当前用药：{esc('、'.join(patient['medications']) or '未填写')}</p>",
                "<h2>症状解析</h2>", table(["标准症状", "状态", "原文证据"],
                    [[s['canonical'], '阳性' if s['present'] else '否定', s['evidence']] for s in payload['parsed_symptoms']]),
                "<h2>证候判断</h2>", table(["证候", "相对支持度", "判断依据"],
                    [[s['name'], f"{s['confidence']:.1%}", s['rationale']] for s in payload['syndromes']]),
                "<h2>中药候选与解释</h2>", table(["序号", "中药", "计算得分（见运行模式）", "证据路径"],
                    [[s['rank'], s['name'], f"{s['score']:.3f}", '；'.join(s['evidence_paths'])] for s in payload['recommendations']]),
                "<h2>约束排除结果</h2>", table(["中药", "原因"],
                    [[h, '；'.join(v)] for h,v in payload['excluded_herbs'].items()]),
                "<h2>运行信息</h2><p>" + esc(json_export(payload.get('metadata', {}))) + "</p>",
                "<h2>数据与使用说明</h2><p>数据范围：" + esc(payload.get('metadata', {}).get('data_scope', '历史记录，数据范围请参照原版本')) + "。相对支持度不代表临床诊断概率；证据覆盖不代表疗效。</p>",
                *[f"<p>{esc(note)}</p>" for note in payload['safety_notes']]]
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DMease辅助分析报告</title><style>' \
        'body{font-family:"Microsoft YaHei",sans-serif;max-width:1000px;margin:36px auto;line-height:1.65;color:#111}' \
        'h1{font-size:24px}h2{font-size:18px;margin-top:28px}table{width:100%;border-collapse:collapse;font-size:13px}' \
        'th,td{border:1px solid #555;padding:8px;text-align:left;overflow-wrap:anywhere}p{white-space:pre-wrap}' \
        '@media print{body{margin:0}tr{break-inside:avoid}h2{break-after:avoid}}' \
        '</style><body>' + ''.join(sections) + '</body></html>'
