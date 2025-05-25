from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import Infer 

app = Flask(__name__)
CSV_PATH = Path("BTL_Mining/data_10_1.csv")

df = pd.read_csv(CSV_PATH)
if "Chẩn Đoán" not in df.columns:
    df.insert(1, "Chẩn Đoán", "")

ORIG_COLS = [c for c in df.columns if c != "Chẩn Đoán"]

def _verdict(raw):
    truthy = {True, 1, "1", "true", "True", "T"}
    return "Nguy Kịch" if raw in truthy else "Tiên lượng tốt"

@app.route("/")
def index():
    return render_template_string(TEMPLATE, rows=df.to_dict("records"), cols=ORIG_COLS)

@app.post("/diagnose")
def diagnose_one():
    row_idx = int(request.form["row"])  # zero‑based
    raw = Infer.infer_from_row(str(CSV_PATH), row_idx)
    verdict = _verdict(raw)
    df.at[row_idx, "Chẩn Đoán"] = verdict
    return jsonify(result=verdict)

TEMPLATE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Chẩn đoán sức khoẻ bệnh nhân</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <style>
    .nguy-kich { color:#dc3545 !important; font-weight:bold; }
  </style>
</head>
<body class="p-4">
  <h3 class="mb-4">📊 Chẩn đoán qua chỉ số</h3>

  <div class="d-flex align-items-center gap-3 mb-3 flex-wrap">
    <button id="diag-all" class="btn btn-danger">🩺 Chẩn đoán TẤT CẢ</button>
    <div class="progress flex-grow-1" style="max-width:320px; height: 24px;">
      <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width:0%">0%</div>
    </div>
  </div>

  <div class="table-responsive">
    <table id="data-table" class="table table-bordered table-sm align-middle text-center">
      <thead class="table-light">
        <tr>
          <th>Hành động</th>
          <th>Chẩn Đoán</th>
          {% for c in cols %}<th>{{ c }}</th>{% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr data-row="{{ loop.index0 }}">
          <td><button class="btn btn-primary btn-sm diagnose-one">Chẩn đoán</button></td>
          <td class="diag-result {% if row["Chẩn Đoán"].strip() == 'Nguy Kịch' %}nguy-kich{% endif %}">{{ row["Chẩn Đoán"] }}</td>
          {% for c in cols %}<td>{{ row[c] }}</td>{% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

<script>
$(function(){
  function updateCellAppearance($cell, verdict){
      const danger = verdict.trim() === "Nguy Kịch";
      $cell.toggleClass("nguy-kich", danger);
  }

  // Chẩn đoán một dòng
  $(document).on("click", ".diagnose-one", function(){
    const $tr   = $(this).closest("tr");
    const idx   = $tr.data("row");
    const $cell = $tr.find(".diag-result");
    $.post("/diagnose", {row: idx}, function(res){
        $cell.text(res.result);
        updateCellAppearance($cell, res.result);
    });
  });

  // Chẩn đoán tất cả với progress bar
  $("#diag-all").on("click", function(){
    const $btn = $(this).prop("disabled", true);
    const $bar = $("#progress-bar");
    const $rows = $("tbody tr");
    const total = $rows.length;
    let completed = 0;

    // Reset progress UI
    $bar.css("width", "0%").text("0%");

    $rows.each(function(){
        const $tr   = $(this);
        const idx   = $tr.data("row");
        const $cell = $tr.find(".diag-result");
        $.post("/diagnose", {row: idx}, function(res){
            $cell.text(res.result);
            updateCellAppearance($cell, res.result);
        }).always(function(){
            completed++;
            const pct = Math.round((completed/total)*100);
            $bar.css("width", pct + "%").text(pct + "%");
            if(completed === total){
                $btn.prop("disabled", false);
            }
        });
    });
  });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
