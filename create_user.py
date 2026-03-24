#!/usr/bin/env python3
"""
Интерактивный скрипт для создания нового пользователя.
Пользователь вводит данные и выбирает интересующие теги из списка.
"""

import sys
from app import app
from models import db, User, Item

def get_unique_tags():
    """Возвращает список уникальных тегов из всех материалов."""
    with app.app_context():
        items = Item.query.all()
        tags_set = set()
        for item in items:
            if item.tags:
                tags_set.update(item.get_tags_list())
        return sorted(list(tags_set))

def select_tags(tags_list):
    """Интерактивный выбор тегов из списка."""
    if not tags_list:
        print("⚠️ В базе пока нет материалов с тегами. Пропускаем выбор навыков.")
        return []

    print("\n📋 Доступные навыки/теги:")
    for i, tag in enumerate(tags_list, 1):
        print(f"  {i}. {tag}")

    print("\nВведите номера тегов через запятую (например: 1,3,5) или оставьте пустым, чтобы пропустить:")
    choice = input("> ").strip()

    if not choice:
        return []

    selected = []
    for part in choice.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(tags_list):
                selected.append(tags_list[idx-1])
            else:
                print(f"⚠️ Номер {idx} вне диапазона, пропущен.")
        else:
            print(f"⚠️ '{part}' не является числом, пропущен.")

    return selected

def get_input(prompt, default=None):
    """Запрашивает ввод с возможным значением по умолчанию."""
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    else:
        return input(f"{prompt}: ").strip()

def main():
    print("=" * 50)
    print("🚀 Создание нового пользователя")
    print("=" * 50)

    # Получаем уникальные теги из материалов
    tags = get_unique_tags()

    with app.app_context():
        # Ввод данных
        name = get_input("Имя", "Новый сотрудник")
        position = get_input("Должность", "Специалист")
        department = get_input("Отдел", "Разработка")
        avatar = get_input("Аватар (эмодзи)", "👤")
        color = get_input("Цвет (HEX, например #3498db)", "#3498db")
        projects_input = get_input("Проекты (через запятую)", "")
        hire_date = get_input("Дата найма (ГГГГ-ММ-ДД)", "2025-01-01")
        role = get_input("Роль (user/admin)", "user")

        # Выбор тегов (навыков) из списка
        selected_tags = select_tags(tags)
        # В модели User нет поля tags, но можно сохранить в projects или игнорировать.
        # Поскольку модель User не хранит теги напрямую, мы можем просто вывести их для информации.
        if selected_tags:
            print(f"\n✅ Выбраны навыки: {', '.join(selected_tags)}")
            # Можно сохранить их, например, в поле projects (дописать)
            if projects_input:
                projects_input += "," + ",".join(selected_tags)
            else:
                projects_input = ",".join(selected_tags)
            print("   (навыки добавлены в поле 'Проекты')")
        else:
            print("Навыки не выбраны.")

        # Создаём пользователя
        user = User(
            name=name,
            position=position,
            department=department,
            avatar=avatar,
            color=color,
            projects=projects_input if projects_input else None,
            hire_date=hire_date,
            role=role
        )

        db.session.add(user)
        db.session.commit()

        print("\n✅ Пользователь успешно создан!")
        print(f"   ID: {user.id}")
        print(f"   Имя: {user.name}")
        print(f"   Должность: {user.position}")
        print(f"   Отдел: {user.department}")
        print(f"   Роль: {user.role}")
        if selected_tags:
            print(f"   Навыки: {', '.join(selected_tags)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Операция отменена.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
