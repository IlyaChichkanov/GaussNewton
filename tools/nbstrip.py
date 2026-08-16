#!/usr/bin/env python3
"""git clean-фильтр: выкидывает из .ipynb тяжёлый вывод, оставляя текстовый.

Читает ноутбук из stdin, пишет очищенный в stdout. Подключается как
    git config filter.nbstrip.clean "python3 tools/nbstrip.py"
плюс строка `*.ipynb filter=nbstrip` в .gitattributes.

Что уходит (99% веса — картинки и plotly-фигуры):
  - image/png, image/jpeg, image/svg+xml, image/gif;
  - application/vnd.plotly.v1+json и прочие vnd.*-виджеты;
  - text/html длиннее HTML_LIMIT (встроенный plotly.js, дампы pandas);
  - metadata.widgets — сохранённое состояние ipywidgets.

Что остаётся: text/plain, stream (логи прогонов, времена итераций, невязки),
traceback ошибок, весь код и markdown. Рабочая копия на диске не меняется —
фильтр работает только на пути в индекс, поэтому графики остаются видны
локально, а `git status` при этом чист.

Преобразование детерминированное: один и тот же ноутбук всегда даёт
побайтово одинаковый результат, иначе git считал бы файл изменённым.
"""
import json
import sys

HTML_LIMIT = 65536          # длиннее — это встроенная библиотека, не таблица

DROP_PREFIXES = ("image/", "application/vnd.")
DROP_EXACT = {"application/javascript"}          # text/latex и text/plain остаются


def _strip_output(output):
    """Вернуть очищенный output или None, если от него ничего не осталось."""
    data = output.get("data")
    if data is None:
        return output                      # stream / error — оставляем целиком

    kept = {}
    for mime, payload in data.items():
        if mime.startswith(DROP_PREFIXES) or mime in DROP_EXACT:
            continue
        if mime == "text/html" and len(json.dumps(payload)) > HTML_LIMIT:
            continue
        kept[mime] = payload

    if not kept:
        return None
    output["data"] = kept
    return output


def strip(nb):
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        cell["outputs"] = [o for o in (_strip_output(o) for o in outputs)
                           if o is not None]
    return nb


def main():
    raw = sys.stdin.read()
    try:
        nb = json.loads(raw)
    except ValueError:
        sys.stdout.write(raw)              # не JSON — пропускаем как есть
        return
    text = json.dumps(strip(nb), indent=1, ensure_ascii=False, sort_keys=False)
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
