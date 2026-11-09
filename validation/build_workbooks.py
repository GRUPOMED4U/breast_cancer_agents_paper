"""Build the clinician review workbooks (#13, #14).

One workbook per oncologist (24 cases: 12 shared + 12 unique, in a random
order per oncologist) plus one calibration workbook (2 cases outside the
sample, for the joint training session). Each case has its own sheet, laid out
in the session order defined in #14:

  Etapa 1  case summary + blinded response -> holistic rating     (visible)
  Etapa 2  reference decision -> classification                 (collapsed)
  Etapa 3  rubric audit, one row per criterion                  (collapsed)
  Etapa 4  met / not met for the Etapa 1 response (last column of the Etapa 3 table)
  Etapa 5  optional: second response, negative criteria only    (collapsed)

Later steps are collapsed row groups so they stay out of sight until the
reviewer opens them. The EHR patient-ID line is removed from summaries.

Hidden column L holds a stable item id on every answer row, so the analysis can
read the completed workbooks back without depending on the layout.

Inputs: validation/output/allocation.csv, validation/output/selected_responses.jsonl
Outputs (gitignored):
  validation/output/workbooks/revisao_oncologista_1.xlsx
  validation/output/workbooks/revisao_oncologista_2.xlsx
  validation/output/workbooks/calibracao.xlsx
  validation/output/answer_key_rubrics.csv

Usage:
    .venv/Scripts/python.exe validation/build_workbooks.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.properties import Outline

from common import SEED, deidentify_summary, ensure_output_dir, load_cases

# ---------------------------------------------------------------------------
# Clinician-facing labels (Portuguese). Edit here to change wording/language.
# ---------------------------------------------------------------------------
CATEGORY = {
    "systemic_therapy": "Terapia sistêmica",
    "radiotherapy": "Radioterapia",
    "surgery": "Cirurgia",
    "complementary_exams": "Exames complementares",
}
OPT = {
    "rating": ["1", "2", "3", "4", "5"],
    "yes_no": ["Sim", "Não"],
    "yes_no_unsure": ["Sim", "Não", "Incerto"],
    "ref_class": ["Concordante com diretrizes", "Alternativa aceitável", "Não concordante", "Incerto"],
    "support": ["Decisão de referência", "Evidência externa / diretriz", "Ambos", "Nenhum", "Incerto"],
    "weight": ["Baixo demais", "Adequado", "Alto demais"],
    "set_ok": ["Sim", "Sim, com ressalvas", "Não"],
    "status": ["Não iniciado", "Em andamento", "Concluído"],
}
RUBRIC_COLS = [
    # (header, option key, item-id suffix)
    ("Etapa 3: Correto?", "yes_no_unsure", "correct"),
    ("Etapa 3: Clinicamente relevante?", "yes_no", "relevant"),
    ("Etapa 3: Fundamentado em", "support", "support"),
    ("Etapa 3: Peso", "weight", "weight"),
    ("Etapa 3: Sinal (+/−) invertido?", "yes_no", "sign_wrong"),
    ("Etapa 3: Penaliza ou deixa de premiar alternativa válida?", "yes_no", "penalizes_alt"),
    ("Etapa 4: Atendido pela resposta da Etapa 1?", "yes_no", "met"),
]
INSTRUCTIONS = """INSTRUÇÕES PARA A REVISÃO

Objetivo: avaliar (1) a qualidade de respostas geradas por sistemas de IA, (2) a decisão clínica de referência e (3) os critérios (rubricas) usados para pontuar automaticamente as respostas.

Cada caso tem sua própria aba (R01, R02...). Use a aba "Índice" para navegar e marcar o andamento. Os casos estão em ordem aleatória.

Siga SEMPRE a ordem das etapas dentro de cada caso. As etapas 2, 3 e 5 estão recolhidas: clique no sinal "+" à esquerda da planilha para abri-las SOMENTE depois de concluir a etapa anterior. Não volte para alterar a Etapa 1 depois de ver a decisão de referência.

Preencha apenas as células amarelas. A maioria tem lista suspensa.

ETAPA 1 — Avaliação global (cega)
Leia o resumo do caso e a resposta do sistema. Você não sabe qual modelo ou arquitetura gerou a resposta; trechos que poderiam identificá-la foram substituídos por [removido]. Algumas respostas estão em inglês e outras em português; isso não deve influenciar a nota.
- Qualidade global (1 = muito ruim / potencialmente danosa; 3 = aceitável com problemas relevantes; 5 = excelente, equivalente ao que um especialista recomendaria).
- Clinicamente aceitável: você estaria confortável se essa recomendação fosse seguida para esta paciente?

ETAPA 2 — Decisão de referência
Leia a decisão registrada pela equipe assistente.
- Concordante com diretrizes: segue diretrizes atuais (ex.: NCCN, ESMO, ASCO).
- Alternativa aceitável: não é a opção preferencial, mas é uma conduta razoável e defensável.
- Não concordante: diverge de diretrizes sem justificativa aparente no resumo.
- Liste alternativas aceitáveis que a referência não contempla.
- Indique se a decisão parece depender de informação que NÃO está no resumo (ex.: preferência da paciente, comorbidade, acesso a medicamento).

ETAPA 3 — Auditoria dos critérios (rubricas)
Cada linha é um critério usado pelo avaliador automático. Pontos positivos premiam a resposta quando o critério é atendido; pontos negativos penalizam (ex.: −5 para "Recomenda quimioterapia" significa que recomendar quimioterapia é indesejado). Os critérios estão em inglês.
- Correto?: o conteúdo do critério está clinicamente correto para este caso?
- Clinicamente relevante?: vale a pena avaliar isso numa recomendação de tratamento?
- Fundamentado em: o critério se apoia na decisão de referência, em evidência externa/diretriz, em ambos ou em nada?
- Peso: a magnitude dos pontos é adequada em relação aos outros critérios do caso?
- Sinal invertido?: o critério premia algo que deveria penalizar, ou vice-versa?
- Penaliza ou deixa de premiar alternativa válida?: o critério prejudica uma resposta que propõe outra conduta aceitável?
Ao final, liste critérios importantes que estão faltando e diga se o conjunto é aceitável para avaliar este caso.

ETAPA 4 — Critério atendido?
Na última coluna da tabela, indique se a resposta da Etapa 1 atende a cada critério (para critérios negativos: "Sim" significa que a resposta FAZ a coisa indesejada).

ETAPA 5 — Opcional
Uma segunda resposta para o mesmo caso. Leia e marque apenas se ela atende a cada critério negativo listado.

Tempo estimado: cerca de 30 minutos por caso (+7 minutos para a Etapa 5).
Dúvidas: registre no campo de comentários do caso.
"""

# ---------------------------------------------------------------------------
FILL_ANSWER = PatternFill("solid", fgColor="FFF2CC")
FILL_HEADER = PatternFill("solid", fgColor="D9E1F2")
FILL_STEP = PatternFill("solid", fgColor="1F3864")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
TEXT_COLS = "B:K"
CHARS_PER_LINE = 170  # approx. characters per line across the merged B:K block
ID_COL = 12  # column L, hidden


class CaseSheet:
    def __init__(self, ws, validations):
        self.ws = ws
        self.row = 1
        self.dv = validations

    def _dv(self, key):
        if key not in self.dv:
            formula = '"' + ",".join(OPT[key]) + '"'
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            dv.error = "Escolha uma opção da lista."
            dv.errorTitle = "Valor inválido"
            self.ws.add_data_validation(dv)
            self.dv[key] = dv
        return self.dv[key]

    def step(self, text):
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(bold=True, color="FFFFFF", size=12)
        for col in range(1, 12):
            self.ws.cell(self.row, col).fill = FILL_STEP
        self.row += 1

    def header(self, text):
        self.ws.merge_cells(start_row=self.row, start_column=1, end_row=self.row, end_column=11)
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(bold=True)
        c.fill = FILL_HEADER
        self.row += 1

    def text_block(self, text):
        for raw in text.splitlines():
            line = raw.strip()
            if not line or set(line) <= {"-", "*", "_"}:
                continue
            bold = line.startswith("#")
            line = re.sub(r"^#+\s*", "", line).replace("**", "")
            self.ws.merge_cells(f"B{self.row}:K{self.row}")
            c = self.ws.cell(self.row, 2, line)
            c.alignment = WRAP
            if bold:
                c.font = Font(bold=True)
            lines = max(1, math.ceil(len(line) / CHARS_PER_LINE))
            self.ws.row_dimensions[self.row].height = min(409, 15 * lines)
            self.row += 1

    def question(self, label, item_id, opt_key=None, free_text=False):
        self.ws.merge_cells(f"B{self.row}:D{self.row}")
        q = self.ws.cell(self.row, 2, label)
        q.alignment = WRAP
        q.font = Font(bold=True)
        if free_text:
            self.ws.merge_cells(f"E{self.row}:K{self.row}")
            self.ws.row_dimensions[self.row].height = 60
        a = self.ws.cell(self.row, 5)
        a.fill = FILL_ANSWER
        a.border = BORDER
        a.alignment = WRAP
        if opt_key:
            self._dv(opt_key).add(a.coordinate)
        self.ws.cell(self.row, ID_COL, item_id)
        self.row += 1

    def rubric_table(self, rubrics, idx_list, cols, id_prefix):
        heads = ["#", "Critério", "Pontos", "Categoria"] + [h for h, _, _ in cols]
        for j, h in enumerate(heads, start=1):
            c = self.ws.cell(self.row, j, h)
            c.font = Font(bold=True)
            c.fill = FILL_HEADER
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            c.border = BORDER
        self.ws.row_dimensions[self.row].height = 48
        self.row += 1
        for n, i in enumerate(idx_list, start=1):
            r = rubrics[i]
            pts = r["points"]
            values = [n, r["criterion"], f"+{pts}" if pts > 0 else str(pts), ", ".join(CATEGORY.get(t, t) for t in r["tags"])]
            for j, v in enumerate(values, start=1):
                c = self.ws.cell(self.row, j, v)
                c.alignment = WRAP
                c.border = BORDER
                if j == 3:
                    c.font = Font(bold=True, color="C00000" if pts < 0 else "006100")
            for j, (_, key, _) in enumerate(cols, start=5):
                c = self.ws.cell(self.row, j)
                c.fill = FILL_ANSWER
                c.border = BORDER
                self._dv(key).add(c.coordinate)
            self.ws.cell(self.row, ID_COL, f"{id_prefix}:{i}")
            self.ws.row_dimensions[self.row].height = min(409, 15 * max(1, math.ceil(len(r["criterion"]) / 75)))
            self.row += 1

    def blank(self, n=1):
        self.row += n

    def group(self, start, end):
        if end >= start:
            self.ws.row_dimensions.group(start, end, outline_level=1, hidden=True)


def build_case_sheet(wb, code, case, primary, topup):
    ws = wb.create_sheet(code)
    ws.sheet_properties.outlinePr = Outline(summaryBelow=False, summaryRight=False)
    widths = {"A": 5, "B": 70, "C": 8, "D": 16}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for col in "EFGHIJK":
        ws.column_dimensions[col].width = 17
    ws.column_dimensions["L"].hidden = True
    s = CaseSheet(ws, {})
    rubrics = case["rubrics"]

    title = ws.cell(1, 1, f"Caso {code}")
    title.font = Font(bold=True, size=14)
    ws.cell(1, ID_COL, f"case_code:{code}")
    s.row = 2
    ws.cell(s.row, 2, "Siga a ordem das etapas. Abra as etapas recolhidas (+ à esquerda) somente após concluir a anterior.").font = Font(italic=True)
    s.blank(2)

    s.step("ETAPA 1 — Avaliação global da resposta (cega)")
    s.header("Resumo do caso")
    s.text_block(deidentify_summary(case["case_summary"]))
    s.blank()
    s.header("Resposta do sistema")
    s.text_block(primary["text"])
    s.blank()
    s.question("Qualidade global da resposta (1–5)", "holistic_rating", "rating")
    s.question("A resposta é clinicamente aceitável?", "holistic_acceptable", "yes_no")
    s.question("Comentários sobre a resposta (opcional)", "holistic_comment", free_text=True)
    s.blank()

    s.step("ETAPA 2 — Decisão de referência (abra após concluir a Etapa 1)")
    start = s.row
    s.header("Decisão registrada pela equipe assistente")
    s.text_block(case["official_recommendations"])
    s.blank()
    s.question("Classificação da decisão de referência", "ref_class", "ref_class")
    s.question("Alternativas aceitáveis não contempladas pela referência", "ref_missing_alternatives", free_text=True)
    s.question("A decisão depende de informação ausente no resumo?", "ref_depends_missing_info", "yes_no_unsure")
    s.question("Se sim, qual informação?", "ref_missing_info_detail", free_text=True)
    s.blank()
    s.group(start, s.row - 1)

    s.step("ETAPA 3 e 4 — Auditoria dos critérios (abra após concluir a Etapa 2)")
    start = s.row
    s.rubric_table(rubrics, list(range(len(rubrics))), RUBRIC_COLS, "rubric")
    s.blank()
    s.question("Critérios importantes que estão faltando", "rubric_missing_criteria", free_text=True)
    s.question("O conjunto de critérios é aceitável para avaliar este caso?", "rubric_set_acceptable", "set_ok")
    s.question("Comentários sobre os critérios (opcional)", "rubric_comment", free_text=True)
    s.blank()
    s.group(start, s.row - 1)

    neg_idx = [i for i, r in enumerate(rubrics) if r["points"] < 0]
    s.step("ETAPA 5 (opcional) — Segunda resposta, apenas critérios negativos")
    start = s.row
    s.header("Segunda resposta do sistema")
    s.text_block(topup["text"])
    s.blank()
    s.rubric_table(rubrics, neg_idx, [("Atendido pela segunda resposta?", "yes_no", "met")], "topup")
    s.blank()
    s.group(start, s.row - 1)
    ws.sheet_view.zoomScale = 110
    return ws


def build_workbook(path, codes, cases, alloc_by_code, responses):
    wb = Workbook()
    ins = wb.active
    ins.title = "Instruções"
    ins.column_dimensions["A"].width = 140
    for i, line in enumerate(INSTRUCTIONS.splitlines(), start=1):
        c = ins.cell(i, 1, line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if line.isupper() or line.startswith("ETAPA"):
            c.font = Font(bold=True)

    idx = wb.create_sheet("Índice")
    for j, (h, w) in enumerate([("Caso", 10), ("Ir para a aba", 18), ("Andamento", 18), ("Observações", 60)], start=1):
        c = idx.cell(1, j, h)
        c.font = Font(bold=True)
        c.fill = FILL_HEADER
        idx.column_dimensions["ABCD"[j - 1]].width = w
    dv = DataValidation(type="list", formula1='"' + ",".join(OPT["status"]) + '"', allow_blank=True)
    idx.add_data_validation(dv)
    for i, code in enumerate(codes, start=2):
        idx.cell(i, 1, code)
        link = idx.cell(i, 2, f"Abrir {code}")
        link.hyperlink = f"#'{code}'!A1"
        link.font = Font(color="0563C1", underline="single")
        st = idx.cell(i, 3, "Não iniciado")
        st.fill = FILL_ANSWER
        dv.add(st.coordinate)
        idx.cell(i, 4).fill = FILL_ANSWER

    for code in codes:
        pid = alloc_by_code[code]["prompt_id"]
        build_case_sheet(wb, code, cases[pid], responses[(code, "primary")], responses[(code, "topup")])
    wb.active = 1
    wb.save(path)


def main() -> None:
    out_dir = ensure_output_dir()
    with open(out_dir / "allocation.csv", encoding="utf-8") as f:
        allocation = list(csv.DictReader(f))
    alloc_by_code = {a["case_code"]: a for a in allocation}
    responses = {}
    with open(out_dir / "selected_responses.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            responses[(r["case_code"], r["slot"])] = r
    cases = load_cases()

    wb_dir = out_dir / "workbooks"
    wb_dir.mkdir(exist_ok=True)
    shared = sorted(a["case_code"] for a in allocation if a["group"] == "shared")
    for n, onc in enumerate(("onc1", "onc2"), start=1):
        codes = shared + sorted(a["case_code"] for a in allocation if a["group"] == onc)
        random.Random(SEED + n).shuffle(codes)
        path = wb_dir / f"revisao_oncologista_{n}.xlsx"
        build_workbook(path, codes, cases, alloc_by_code, responses)
        print(f"Wrote {path} ({len(codes)} cases)")
    calib = sorted(a["case_code"] for a in allocation if a["group"] == "calibration")
    path = wb_dir / "calibracao.xlsx"
    build_workbook(path, calib, cases, alloc_by_code, responses)
    print(f"Wrote {path} ({len(calib)} cases)")

    # Rubric-level answer key for the analysis (judge verdicts per criterion).
    with open(out_dir / "answer_key_rubrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_code", "prompt_id", "group", "rubric_idx", "points", "category", "judge_met_primary", "judge_met_topup", "criterion"])
        for a in sorted(allocation, key=lambda a: a["case_code"]):
            code, pid = a["case_code"], a["prompt_id"]
            for i, r in enumerate(cases[pid]["rubrics"]):
                writer.writerow([
                    code, pid, a["group"], i, r["points"], "|".join(r["tags"]),
                    responses[(code, "primary")]["judge_met"][i],
                    responses[(code, "topup")]["judge_met"][i] if r["points"] < 0 else "",
                    r["criterion"],
                ])
    print(f"Wrote {out_dir / 'answer_key_rubrics.csv'}")


if __name__ == "__main__":
    main()
