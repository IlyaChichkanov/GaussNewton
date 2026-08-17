#!/usr/bin/env bash
# Локальная настройка репозитория после клонирования.
#
# Зачем это нужно. В .gitattributes объявлен `*.ipynb filter=nbstrip`, но сам
# фильтр — это локальная настройка git, она НЕ хранится в репозитории. На свежем
# клоне git просто не находит фильтр и молча пишет ноутбуки как есть, поэтому
# тяжёлые выводы (картинки, встроенный plotly.js) возвращаются в историю.
#
# Скрипт регистрирует фильтр и помечает его required=true, чтобы отсутствие
# фильтра было явной ошибкой, а не тишиной.
#
# Запуск:  bash tools/setup_repo.sh
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"

git config filter.nbstrip.clean "$python_bin tools/nbstrip.py"
# smudge=cat: рабочая копия не меняется, графики остаются видны локально
git config filter.nbstrip.smudge "cat"
git config filter.nbstrip.required true

echo "git-фильтр nbstrip настроен:"
git config --get-regexp '^filter\.nbstrip\.' | sed 's/^/  /'

echo
echo "Проверка: ноутбуки в индексе теперь без тяжёлого вывода."
echo "Если git status показывает изменения в *.ipynb сразу после настройки —"
echo "это ожидаемо: выводы вычищаются на пути в индекс."
