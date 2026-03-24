from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100))
    department = db.Column(db.String(100))
    avatar = db.Column(db.String(10), default='👤')
    color = db.Column(db.String(7), default='#3498db')
    projects = db.Column(db.String(200))
    hire_date = db.Column(db.String(20))
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Поля для аутентификации - пароль в открытом виде
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)  # Пароль в открытом виде
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'username': self.username,
            'position': self.position,
            'department': self.department,
            'avatar': self.avatar,
            'color': self.color,
            'projects': self.projects.split(',') if self.projects else [],
            'hire_date': self.hire_date,
            'role': self.role,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M')
        }
    
    def to_dict_admin(self):
        """Расширенная версия для админа с паролем"""
        data = self.to_dict()
        data['password'] = self.password  # Показываем пароль админу
        return data

class Item(db.Model):
    __tablename__ = 'items'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50))
    type_icon = db.Column(db.String(10), default='📄')
    tags = db.Column(db.String(200))  # comma-separated
    date = db.Column(db.String(20))
    department = db.Column(db.String(100))
    department_owner = db.Column(db.String(100))
    preview = db.Column(db.Text)
    content = db.Column(db.Text)
    views = db.Column(db.Integer, default=0)
    saved_count = db.Column(db.Integer, default=0)
    author = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_tags_list(self):
        return [t.strip() for t in self.tags.split(',')] if self.tags else []

    def set_tags_list(self, tags_list):
        self.tags = ','.join(tags_list)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'type': self.type,
            'type_icon': self.type_icon,
            'tags': self.get_tags_list(),
            'date': self.date,
            'department': self.department,
            'department_owner': self.department_owner,
            'preview': self.preview,
            'content': self.content,
            'views': self.views,
            'saved_count': self.saved_count,
            'author': self.author
        }

class Interaction(db.Model):
    __tablename__ = 'interactions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
    action = db.Column(db.String(20))  # 'view', 'read', 'save', 'share'
    weight = db.Column(db.Float, default=1.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='interactions')
    item = db.relationship('Item', backref='interactions')
