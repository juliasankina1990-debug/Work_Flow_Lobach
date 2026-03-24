#!/usr/bin/env python3
"""
Скрипт для миграции существующих пользователей:
добавляет username и временные пароли
"""
from app import app
from models import db, User

def migrate_users():
    with app.app_context():
        users = User.query.all()
        migrated_count = 0
        
        for user in users:
            changed = False
            
            # Если нет username, создаем из имени
            if not user.username:
                # Генерируем username из имени
                base_username = user.name.lower().replace(' ', '_')
                username = base_username
                counter = 1
                
                # Проверяем уникальность
                while User.query.filter_by(username=username).first():
                    username = f"{base_username}_{counter}"
                    counter += 1
                
                user.username = username
                print(f"Установлен username '{username}' для пользователя {user.name}")
                changed = True
            
            # Если нет пароля, устанавливаем временный
            if not user.password:
                user.password = 'password123'  # Временный пароль
                print(f"Установлен временный пароль для пользователя {user.name}")
                changed = True
            
            if changed:
                migrated_count += 1
        
        db.session.commit()
        print(f"Мигрировано {migrated_count} пользователей")

if __name__ == '__main__':
    migrate_users()
