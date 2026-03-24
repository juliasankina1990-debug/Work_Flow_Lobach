#!/usr/bin/env python3
"""
Скрипт для создания администратора
Логин: admin
Пароль: qwerty
"""

from app import app
from models import db, User

def create_admin():
    with app.app_context():
        print("=" * 50)
        print("Создание администратора")
        print("=" * 50)
        
        # Проверяем, не существует ли уже пользователь с username='admin'
        existing_admin = User.query.filter_by(username='admin').first()
        
        if existing_admin:
            # Если существует, обновляем пароль и роль
            existing_admin.role = 'admin'
            existing_admin.password = 'qwerty'
            print(f"Обновлен существующий пользователь:")
            print(f"  - Имя: {existing_admin.name}")
            print(f"  - Логин: admin")
            print(f"  - Пароль: qwerty")
        else:
            # Создаем нового админа
            new_admin = User(
                username='admin',
                name='Главный администратор',
                password='qwerty',  # Пароль в открытом виде
                position='System Administrator',
                department='IT',
                avatar='👑',
                color='#ff0000',
                projects='Administration',
                hire_date='2024',
                role='admin'
            )
            
            db.session.add(new_admin)
            print(f"Создан новый администратор:")
            print(f"  - Имя: Главный администратор")
            print(f"  - Логин: admin")
            print(f"  - Пароль: qwerty")
        
        db.session.commit()
        
        print("=" * 50)
        print("Готово!")
        print("=" * 50)
        print("Данные для входа:")
        print("  Логин: admin")
        print("  Пароль: qwerty")
        print("=" * 50)

if __name__ == '__main__':
    create_admin()
