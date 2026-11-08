"""Дозапись ячеек режима C, обнулённых из-за префикса 'arXiv:' в id.

Пока живой процесс eval_benchmark пишется в eval_results.json, файл не трогаем —
скрипт это проверяет и отказывается. Режимы:
  python zero_cells.py            -- показать
  python zero_cells.py --purge    -- резервная копия + удалить обнулённые ячейки, чтобы следующий запуск их пересчитал
"""
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
MODE_C = "mode_c_papertrace"


def is_run_alive() -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True).stdout
    return "eval_benchmark" in out


def load():
    with open(RESULTS, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, data
    for key, value in data.items():
        if isinstance(value, list):
            return data, value
    raise SystemExit("в eval_results.json нет списка ячеек")


def main():
    purge = "--purge" in sys.argv
    if is_run_alive():
        raise SystemExit("живой прогон не завершён — файл не трогаем")
    data, rows = load()
    bad = [r for r in rows if r.get("mode") == MODE_C and not r.get("total_claims")]
    print(f"ячеек всего {len(rows)}, обнулённых C {len(bad)}:")
    for r in bad:
        print("  ", r["question_id"], r.get("latency_sec"), r.get("total_tokens"))
    if not purge or not bad:
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = RESULTS + ".bak-" + stamp
    shutil.copy2(RESULTS, backup)
    print("резервная копия:", backup)
    keep = [r for r in rows if not (r.get("mode") == MODE_C and not r.get("total_claims"))]
    if isinstance(data, list):
        payload = keep
    else:
        for key, value in list(data.items()):
            if value is rows:
                data[key] = keep
                break
        payload = data
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, RESULTS)
    print(f"удалено {len(rows) - len(keep)} ячеек, осталось {len(keep)} — следующий запуск их пересчитает")
    for path in sorted(glob.glob(RESULTS + ".bak-*"))[:-3]:
        print("старые копии (можно удалить вручную):", path)


if __name__ == "__main__":
    main()
