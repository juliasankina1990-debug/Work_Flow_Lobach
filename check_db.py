#!/usr/bin/env python3
"""
Скрипт для проверки структуры таблиц и добавления недостающих колонок
"""
from app import app
from models import db, User
from sqlalchemy import inspect, text

def check_and_fix_database():
    with app.app_context():
        inspector = inspect(db.engine)
        
        # Проверяем наличие колонок в таблице users
        columns = [col['name'] for col in inspector.get_columns('users')]
        
        print("Существующие колонки в таблице users:", columns)
        
        # Добавляем недостающие колонки
        if 'username' not in columns:
            print("Добавляем колонку username...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN username VARCHAR(50) UNIQUE'))
                conn.commit()
            print("Колонка username добавлена")
        
        if 'password' not in columns:
            print("Добавляем колонку password...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN password VARCHAR(50)'))
                conn.commit()
            print("Колонка password добавлена")
        elif 'password_hash' in columns:
            # Если есть password_hash, переименовываем или удаляем
            print("Обнаружена колонка password_hash, создаем password...")
            with db.engine.connect() as conn:
                # Сначала добавляем новую колонку
                conn.execute(text('ALTER TABLE users ADD COLUMN password VARCHAR(50)'))
                # Копируем данные (если нужно)
                conn.execute(text('UPDATE users SET password = "password123" WHERE password IS NULL'))
                conn.commit()
            print("Колонка password добавлена")
        
        print("Проверка структуры базы данных завершена")

if __name__ == '__main__':
    check_and_fix_database()
