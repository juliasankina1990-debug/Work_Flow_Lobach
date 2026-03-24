#!/bin/bash

# Скрипт для вывода содержимого файлов проекта
# Показывает только файлы с указанными расширениями
# Исключает директории: __pycache__, backups, model_metrics, venv

# Укажите нужные расширения через |
EXTENSIONS="\.py$|\.js$|\.html$|\.css$|\.json$|\.yaml$|\.yml$|\.txt$|\.md$|\.sh$|\.sql$|\.xml$"

echo "=== Содержимое проекта (только текстовые файлы) ==="
echo ""

# Находим файлы с нужными расширениями, исключая ненужные директории
find . \
    -type d -name "__pycache__" -prune -o \
    -type d -name "backups" -prune -o \
    -type d -name "model_metrics" -prune -o \
    -type d -name "venv" -prune -o \
    -type d -name ".git" -prune -o \
    -type f | while read -r file; do
    
    # Проверяем расширение файла
    if echo "$file" | grep -qE "$EXTENSIONS"; then
        # Пропускаем сам скрипт
        if [[ "$file" == "./cat_project.sh" ]]; then
            continue
        fi
        
        echo "========================================="
        echo "Файл: $file"
        echo "========================================="
        cat "$file"
        echo ""
        echo ""
    fi
done

echo "=== Конец содержимого проекта ==="
