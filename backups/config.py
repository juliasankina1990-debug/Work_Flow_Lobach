import os

class Config:
    # Настройки базы данных
    DB_USER = os.getenv('DB_USER', 'workflow_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '2280')  # Замените на свой пароль
    DB_NAME = os.getenv('DB_NAME', 'workflow_db')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')

    SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
