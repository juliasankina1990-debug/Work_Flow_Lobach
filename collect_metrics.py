#!/usr/bin/env python3
"""
Скрипт для сбора метрик работы ИИ-рекомендательной системы
и построения графиков для курсовой работы.

Запуск: python3 collect_metrics.py
Результат: создаст папку 'ai_metrics' с графиками и таблицами
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Без отображения графиков
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import json
import traceback

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import db, User, Item, Interaction
from recommendation import recommender

# Настройка стилей графиков
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class AIMetricsCollector:
    def __init__(self):
        self.output_dir = 'ai_metrics'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics = {}
        
    def setup_output_dir(self):
        """Создает папку для сохранения результатов"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Создаем подпапки
        for subdir in ['tables', 'graphs']:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        print(f"📁 Результаты будут сохранены в папку: {self.output_dir}")
    
    def collect_basic_stats(self):
        """Собирает базовую статистику системы"""
        print("\n" + "="*60)
        print("📊 СБОР БАЗОВОЙ СТАТИСТИКИ")
        print("=*60")
        
        with app.app_context():
            users = User.query.all()
            items = Item.query.all()
            interactions = Interaction.query.all()
            
            # Фильтруем по типам взаимодействий
            views = Interaction.query.filter_by(action='view').count()
            reads = Interaction.query.filter_by(action='read').count()
            saves = Interaction.query.filter_by(action='save').count()
            shares = Interaction.query.filter_by(action='share').count()
            
            stats = {
                'total_users': len(users),
                'total_materials': len(items),
                'total_interactions': len(interactions),
                'views': views,
                'reads': reads,
                'saves': saves,
                'shares': shares,
                'avg_views_per_material': round(views / len(items) if items else 0, 2),
                'avg_reads_per_user': round(reads / len(users) if users else 0, 2),
            }
            
            # Считаем по ролям
            roles = {}
            for user in users:
                roles[user.role] = roles.get(user.role, 0) + 1
            stats['users_by_role'] = roles
            
            # Считаем по типам материалов
            types = {}
            for item in items:
                if item.type:
                    types[item.type] = types.get(item.type, 0) + 1
            stats['materials_by_type'] = types
            
            print(f"✅ Пользователей: {stats['total_users']}")
            print(f"✅ Материалов: {stats['total_materials']}")
            print(f"✅ Взаимодействий: {stats['total_interactions']}")
            print(f"   - Просмотры: {stats['views']}")
            print(f"   - Прочтения: {stats['reads']}")
            print(f"   - Сохранения: {stats['saves']}")
            print(f"   - Поделились: {stats['shares']}")
            
            self.metrics['basic_stats'] = stats
            return stats
    
    def collect_user_activity_metrics(self):
        """Анализирует активность пользователей"""
        print("\n" + "="*60)
        print("👥 АНАЛИЗ АКТИВНОСТИ ПОЛЬЗОВАТЕЛЕЙ")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()
            user_activity = []
            
            for user in users:
                user_interactions = Interaction.query.filter_by(user_id=user.id).all()
                user_views = len([i for i in user_interactions if i.action == 'view'])
                user_reads = len([i for i in user_interactions if i.action == 'read'])
                user_saves = len([i for i in user_interactions if i.action == 'save'])
                
                # Получаем теги, которые интересуют пользователя
                read_items = Item.query.join(Interaction).filter(
                    Interaction.user_id == user.id,
                    Interaction.action == 'read'
                ).all()
                
                tags = []
                for item in read_items:
                    tags.extend(item.get_tags_list())
                
                tag_counter = Counter(tags)
                top_tags = [tag for tag, _ in tag_counter.most_common(3)]
                
                user_activity.append({
                    'user_id': user.id,
                    'username': user.username,
                    'role': user.role,
                    'total_interactions': len(user_interactions),
                    'views': user_views,
                    'reads': user_reads,
                    'saves': user_saves,
                    'top_tags': ', '.join(top_tags) if top_tags else 'нет тегов',
                    'interaction_score': user_reads * 3 + user_saves * 5
                })
            
            # Сортируем по активности
            user_activity.sort(key=lambda x: x['interaction_score'], reverse=True)
            
            # Вычисляем статистику
            interactions_list = [u['total_interactions'] for u in user_activity]
            
            metrics = {
                'most_active_users': user_activity[:5],
                'least_active_users': user_activity[-5:] if len(user_activity) >= 5 else user_activity,
                'avg_interactions_per_user': round(np.mean(interactions_list) if interactions_list else 0, 2),
                'median_interactions': round(float(np.median(interactions_list)) if interactions_list else 0, 2),
                'max_interactions': max(interactions_list) if interactions_list else 0,
                'min_interactions': min(interactions_list) if interactions_list else 0,
                'users_with_zero_activity': len([u for u in user_activity if u['total_interactions'] == 0])
            }
            
            print(f"✅ Среднее взаимодействий на пользователя: {metrics['avg_interactions_per_user']:.1f}")
            print(f"✅ Медиана: {metrics['median_interactions']:.1f}")
            print(f"✅ Максимум: {metrics['max_interactions']}")
            print(f"✅ Минимум: {metrics['min_interactions']}")
            print(f"✅ Пользователей без активности: {metrics['users_with_zero_activity']}")
            
            print("\n🏆 Топ-5 активных пользователей:")
            for i, user in enumerate(metrics['most_active_users'], 1):
                print(f"   {i}. {user['username']} (роль: {user['role']}) - {user['total_interactions']} действий")
            
            self.metrics['user_activity'] = metrics
            self.metrics['user_activity_details'] = user_activity
            return metrics
    
    def collect_material_popularity(self):
        """Анализирует популярность материалов"""
        print("\n" + "="*60)
        print("📈 АНАЛИЗ ПОПУЛЯРНОСТИ МАТЕРИАЛОВ")
        print("="*60)
        
        with app.app_context():
            items = Item.query.all()
            material_stats = []
            
            for item in items:
                item_interactions = Interaction.query.filter_by(item_id=item.id).all()
                item_views = len([i for i in item_interactions if i.action == 'view'])
                item_reads = len([i for i in item_interactions if i.action == 'read'])
                item_saves = len([i for i in item_interactions if i.action == 'save'])
                
                # Считаем уникальных пользователей
                unique_users = set([i.user_id for i in item_interactions])
                
                # Сокращаем длинные названия
                short_title = item.title[:40] + '...' if len(item.title) > 40 else item.title
                
                material_stats.append({
                    'item_id': item.id,
                    'title': short_title,
                    'type': item.type or 'без типа',
                    'department': item.department or 'без отдела',
                    'views': item_views,
                    'reads': item_reads,
                    'saves': item_saves,
                    'unique_users': len(unique_users),
                    'popularity_score': item_views + item_reads * 3 + item_saves * 5,
                    'tags': item.tags or ''
                })
            
            # Сортируем по популярности
            material_stats.sort(key=lambda x: x['popularity_score'], reverse=True)
            
            # Вычисляем статистику
            views_list = [m['views'] for m in material_stats]
            reads_list = [m['reads'] for m in material_stats]
            saves_list = [m['saves'] for m in material_stats]
            
            metrics = {
                'most_popular': material_stats[:5],
                'least_popular': material_stats[-5:] if len(material_stats) >= 5 else material_stats,
                'avg_views': round(np.mean(views_list), 2),
                'avg_reads': round(np.mean(reads_list), 2),
                'avg_saves': round(np.mean(saves_list), 2),
                'max_views': max(views_list) if views_list else 0,
                'max_reads': max(reads_list) if reads_list else 0,
                'max_saves': max(saves_list) if saves_list else 0,
                'materials_with_zero_views': len([m for m in material_stats if m['views'] == 0]),
                'total_materials': len(material_stats)
            }
            
            print(f"✅ Среднее просмотров: {metrics['avg_views']:.1f}")
            print(f"✅ Среднее прочтений: {metrics['avg_reads']:.1f}")
            print(f"✅ Среднее сохранений: {metrics['avg_saves']:.1f}")
            print(f"✅ Материалов без просмотров: {metrics['materials_with_zero_views']}")
            
            print("\n🔥 Топ-5 популярных материалов:")
            for i, item in enumerate(metrics['most_popular'], 1):
                print(f"   {i}. {item['title']} - {item['views']} просмотров, {item['reads']} прочтений")
            
            self.metrics['material_popularity'] = metrics
            self.metrics['material_stats_details'] = material_stats
            return metrics
    
    def collect_temporal_metrics(self):
        """Анализирует активность во времени"""
        print("\n" + "="*60)
        print("⏰ АНАЛИЗ АКТИВНОСТИ ПО ВРЕМЕНИ")
        print("="*60)
        
        with app.app_context():
            interactions = Interaction.query.all()
            
            # Группируем по дням
            daily_counts = {}
            hourly_counts = {i: 0 for i in range(24)}
            weekday_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
            
            for inter in interactions:
                if inter.created_at:
                    date_str = inter.created_at.strftime('%Y-%m-%d')
                    daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
                    
                    hour = inter.created_at.hour
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
                    
                    weekday = inter.created_at.weekday()
                    weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1
            
            # Сортируем по дате и берем последние 30 дней
            sorted_dates = sorted(daily_counts.items())
            last_30 = sorted_dates[-30:] if len(sorted_dates) > 30 else sorted_dates
            
            # Названия дней недели
            weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
            weekday_activity = {weekday_names[i]: weekday_counts.get(i, 0) for i in range(7)}
            
            # Находим пиковые значения
            peak_hour = max(hourly_counts, key=hourly_counts.get)
            peak_day = max(weekday_counts, key=weekday_counts.get)
            
            metrics = {
                'daily_activity': dict(last_30),
                'hourly_activity': hourly_counts,
                'weekday_activity': weekday_activity,
                'peak_hour': peak_hour,
                'peak_day': weekday_names[peak_day],
                'total_days_with_activity': len(daily_counts)
            }
            
            print(f"✅ Всего дней с активностью: {metrics['total_days_with_activity']}")
            print(f"✅ Пиковый час активности: {metrics['peak_hour']}:00")
            print(f"✅ Самый активный день: {metrics['peak_day']}")
            
            self.metrics['temporal_metrics'] = metrics
            return metrics
    
    def collect_recommendation_metrics(self):
        """Собирает метрики работы рекомендательной системы"""
        print("\n" + "="*60)
        print("🤖 АНАЛИЗ РАБОТЫ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
        print("="*60)
        
        with app.app_context():
            # Проверяем загружена ли модель
            model_loaded = recommender.model is not None
            print(f"✅ Модель загружена: {model_loaded}")
            
            users = User.query.all()
            recommendation_stats = []
            explanation_stats = []
            total_recs_analyzed = 0
            
            # Анализируем для всех пользователей (но не более 20 для скорости)
            for user in users[:20]:
                try:
                    # Получаем рекомендации
                    recs = recommender.get_recommendations(user.id, n=5)
                    
                    if recs:
                        total_recs_analyzed += len(recs)
                        
                        # Для каждой рекомендации получаем объяснение
                        for item in recs:
                            exp = recommender.explain_recommendation(user.id, item.id)
                            explanation_stats.append({
                                'user_id': user.id,
                                'username': user.username,
                                'item_id': item.id,
                                'item_title': item.title[:30] + '...' if len(item.title) > 30 else item.title,
                                'explanation': exp
                            })
                        
                        # Проверяем, есть ли пересечения с реальными интересами
                        user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
                        read_item_ids = [i.item_id for i in user_reads]
                        
                        relevant_recs = len([r for r in recs if r.id in read_item_ids])
                        
                        recommendation_stats.append({
                            'user_id': user.id,
                            'username': user.username,
                            'total_recs': len(recs),
                            'relevant_recs': relevant_recs,
                            'relevance_rate': round(relevant_recs / len(recs) * 100, 1) if recs else 0
                        })
                    
                except Exception as e:
                    print(f"⚠️ Ошибка для пользователя {user.id}: {str(e)[:50]}")
                    continue
            
            # Собираем статистику по объяснениям
            explanation_types = []
            for exp in explanation_stats:
                exp_text = exp['explanation'].lower()
                if 'совпадают' in exp_text or 'совпадает' in exp_text:
                    explanation_types.append('По совпадению признаков')
                elif 'отдел' in exp_text:
                    explanation_types.append('По отделу')
                elif 'популярн' in exp_text:
                    explanation_types.append('По популярности')
                else:
                    explanation_types.append('Общие предпочтения')
            
            exp_counter = Counter(explanation_types)
            
            # Вычисляем среднюю релевантность
            if recommendation_stats:
                avg_relevance = np.mean([r['relevance_rate'] for r in recommendation_stats])
            else:
                avg_relevance = 0
            
            metrics = {
                'avg_relevance_rate': round(avg_relevance, 1),
                'model_active': model_loaded,
                'total_users_analyzed': len(recommendation_stats),
                'total_recommendations_analyzed': total_recs_analyzed,
                'explanation_distribution': dict(exp_counter),
                'sample_recommendations': explanation_stats[:10]  # Первые 10 для примера
            }
            
            print(f"✅ Средняя релевантность рекомендаций: {metrics['avg_relevance_rate']}%")
            print(f"✅ Проанализировано пользователей: {metrics['total_users_analyzed']}")
            print(f"✅ Проанализировано рекомендаций: {metrics['total_recommendations_analyzed']}")
            
            print("\n📝 Распределение типов объяснений:")
            for exp_type, count in exp_counter.items():
                if explanation_stats:
                    print(f"   - {exp_type}: {count} ({count/len(explanation_stats)*100:.1f}%)")
            
            self.metrics['recommendation_metrics'] = metrics
            self.metrics['recommendation_details'] = recommendation_stats
            self.metrics['explanation_details'] = explanation_stats
            return metrics
    
    def save_tables(self):
        """Сохраняет все данные в CSV таблицы"""
        print("\n" + "="*60)
        print("💾 СОХРАНЕНИЕ ТАБЛИЦ")
        print("="*60)
        
        tables_dir = os.path.join(self.output_dir, 'tables')
        
        # Сохраняем базовую статистику
        if 'basic_stats' in self.metrics:
            df_basic = pd.DataFrame([self.metrics['basic_stats']])
            df_basic.to_csv(os.path.join(tables_dir, '01_basic_stats.csv'), index=False, encoding='utf-8')
            print("✅ 01_basic_stats.csv - Базовая статистика")
        
        # Сохраняем активность пользователей
        if 'user_activity_details' in self.metrics:
            df_users = pd.DataFrame(self.metrics['user_activity_details'])
            df_users.to_csv(os.path.join(tables_dir, '02_user_activity.csv'), index=False, encoding='utf-8')
            print("✅ 02_user_activity.csv - Активность пользователей")
        
        # Сохраняем статистику материалов
        if 'material_stats_details' in self.metrics:
            df_materials = pd.DataFrame(self.metrics['material_stats_details'])
            df_materials.to_csv(os.path.join(tables_dir, '03_material_stats.csv'), index=False, encoding='utf-8')
            print("✅ 03_material_stats.csv - Статистика материалов")
        
        # Сохраняем объяснения
        if 'explanation_details' in self.metrics:
            df_explanations = pd.DataFrame(self.metrics['explanation_details'])
            if not df_explanations.empty:
                df_explanations.to_csv(os.path.join(tables_dir, '04_explanations.csv'), index=False, encoding='utf-8')
                print("✅ 04_explanations.csv - Объяснения рекомендаций")
        
        # Сохраняем временные метрики
        if 'temporal_metrics' in self.metrics:
            # Дневная активность
            df_daily = pd.DataFrame([
                {'date': k, 'interactions': v} 
                for k, v in self.metrics['temporal_metrics']['daily_activity'].items()
            ])
            df_daily.to_csv(os.path.join(tables_dir, '05_daily_activity.csv'), index=False, encoding='utf-8')
            
            # Часовая активность
            df_hourly = pd.DataFrame([
                {'hour': k, 'interactions': v}
                for k, v in self.metrics['temporal_metrics']['hourly_activity'].items()
            ])
            df_hourly.to_csv(os.path.join(tables_dir, '06_hourly_activity.csv'), index=False, encoding='utf-8')
            
            print("✅ 05-06_temporal_activity.csv - Временная активность")
    
    def generate_all_graphs(self):
        """Генерирует все возможные графики"""
        print("\n" + "="*60)
        print("📈 ГЕНЕРАЦИЯ ГРАФИКОВ")
        print("="*60)
        
        graphs_dir = os.path.join(self.output_dir, 'graphs')
        
        # График 1: Распределение типов взаимодействий
        if 'basic_stats' in self.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats = self.metrics['basic_stats']
            actions = ['Просмотры', 'Прочтения', 'Сохранения', 'Поделились']
            counts = [stats['views'], stats['reads'], stats['saves'], stats['shares']]
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            
            bars = ax.bar(actions, counts, color=colors)
            ax.set_title('Распределение типов взаимодействий', fontsize=16, pad=20)
            ax.set_ylabel('Количество', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '01_interaction_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 01_interaction_distribution.png - Распределение взаимодействий")
        
        # График 2: Топ-10 активных пользователей
        if 'user_activity_details' in self.metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            users_df = pd.DataFrame(self.metrics['user_activity_details'])
            top_users = users_df.nlargest(10, 'total_interactions')
            
            x = range(len(top_users))
            width = 0.25
            
            ax.bar([i - width for i in x], top_users['views'], width, label='Просмотры', color='#3498db')
            ax.bar(x, top_users['reads'], width, label='Прочтения', color='#2ecc71')
            ax.bar([i + width for i in x], top_users['saves'], width, label='Сохранения', color='#f39c12')
            
            ax.set_xlabel('Пользователи', fontsize=12)
            ax.set_ylabel('Количество действий', fontsize=12)
            ax.set_title('Топ-10 самых активных пользователей', fontsize=16, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(top_users['username'].values, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '02_top_users.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 02_top_users.png - Топ активных пользователей")
        
        # График 3: Топ-10 популярных материалов
        if 'material_stats_details' in self.metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            materials_df = pd.DataFrame(self.metrics['material_stats_details'])
            top_materials = materials_df.nlargest(10, 'popularity_score')
            
            y_pos = np.arange(len(top_materials))
            ax.barh(y_pos, top_materials['popularity_score'], color='#9b59b6')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_materials['title'].values)
            ax.set_xlabel('Популярность (скор)', fontsize=12)
            ax.set_title('Топ-10 самых популярных материалов', fontsize=16, pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '03_top_materials.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 03_top_materials.png - Топ популярных материалов")
        
        # График 4: Активность по часам
        if 'temporal_metrics' in self.metrics:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # По часам
            hours = list(self.metrics['temporal_metrics']['hourly_activity'].keys())
            hour_counts = list(self.metrics['temporal_metrics']['hourly_activity'].values())
            
            ax1.bar(hours, hour_counts, color='#e67e22')
            ax1.set_xlabel('Час дня', fontsize=12)
            ax1.set_ylabel('Количество взаимодействий', fontsize=12)
            ax1.set_title('Активность по часам', fontsize=14, pad=15)
            ax1.set_xticks(range(0, 24, 2))
            ax1.grid(True, alpha=0.3)
            
            # По дням недели
            days = list(self.metrics['temporal_metrics']['weekday_activity'].keys())
            day_counts = list(self.metrics['temporal_metrics']['weekday_activity'].values())
            
            colors = ['#2ecc71' if i < 5 else '#e74c3c' for i in range(7)]
            ax2.bar(days, day_counts, color=colors)
            ax2.set_xlabel('День недели', fontsize=12)
            ax2.set_ylabel('Количество взаимодействий', fontsize=12)
            ax2.set_title('Активность по дням недели', fontsize=14, pad=15)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('Временной анализ активности', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '04_temporal_activity.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 04_temporal_activity.png - Временная активность")
        
        # График 5: Распределение типов объяснений
        if 'recommendation_metrics' in self.metrics and self.metrics['recommendation_metrics']['explanation_distribution']:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            exp_dist = self.metrics['recommendation_metrics']['explanation_distribution']
            labels = list(exp_dist.keys())
            sizes = list(exp_dist.values())
            
            if sizes:
                colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors[:len(labels)], startangle=90)
                ax.set_title('Типы объяснений рекомендаций ИИ', fontsize=16, pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '05_explanation_types.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 05_explanation_types.png - Типы объяснений")
        
        # График 6: Релевантность рекомендаций
        if 'recommendation_details' in self.metrics and self.metrics['recommendation_details']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            rec_df = pd.DataFrame(self.metrics['recommendation_details'])
            if not rec_df.empty:
                relevance_rates = rec_df['relevance_rate']
                
                ax.hist(relevance_rates, bins=8, color='#27ae60', edgecolor='white', alpha=0.7)
                ax.axvline(np.mean(relevance_rates), color='red', linestyle='--', 
                           linewidth=2, label=f'Средняя: {np.mean(relevance_rates):.1f}%')
                ax.set_xlabel('Релевантность рекомендаций (%)', fontsize=12)
                ax.set_ylabel('Количество пользователей', fontsize=12)
                ax.set_title('Распределение релевантности рекомендаций', fontsize=16, pad=20)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '06_relevance_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 06_relevance_distribution.png - Релевантность рекомендаций")
        
        # График 7: Топ тегов
        if 'material_stats_details' in self.metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Собираем все теги
            all_tags = []
            for item in self.metrics['material_stats_details']:
                if item['tags']:
                    tags = item['tags'].split(',')
                    all_tags.extend([t.strip() for t in tags if t.strip()])
            
            if all_tags:
                tag_counter = Counter(all_tags)
                top_tags = tag_counter.most_common(10)
                
                tags = [t[0] for t in top_tags]
                counts = [t[1] for t in top_tags]
                
                bars = ax.barh(tags, counts, color='#1abc9c')
                ax.set_xlabel('Количество материалов', fontsize=12)
                ax.set_title('Топ-10 самых популярных тегов', fontsize=16, pad=20)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Добавляем значения
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(bar.get_width() + 0.1, i, str(count), va='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '07_top_tags.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 07_top_tags.png - Топ тегов")
        
        # График 8: Динамика активности (последние 30 дней)
        if 'temporal_metrics' in self.metrics and self.metrics['temporal_metrics']['daily_activity']:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            daily = self.metrics['temporal_metrics']['daily_activity']
            dates = list(daily.keys())
            counts = list(daily.values())
            
            if len(dates) > 1:
                ax.plot(dates, counts, marker='o', linestyle='-', color='#e74c3c', linewidth=2, markersize=4)
                ax.fill_between(dates, counts, alpha=0.2, color='#e74c3c')
                ax.set_xlabel('Дата', fontsize=12)
                ax.set_ylabel('Количество взаимодействий', fontsize=12)
                ax.set_title('Динамика активности за последние 30 дней', fontsize=16, pad=20)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Добавляем линию тренда
                if len(counts) > 1:
                    z = np.polyfit(range(len(counts)), counts, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(range(len(counts))), '--', color='blue', alpha=0.7, linewidth=2, label='Тренд')
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '08_activity_trend.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 08_activity_trend.png - Тренд активности")
    
    def generate_report(self):
        """Генерирует итоговый отчет в формате JSON"""
        report_path = os.path.join(self.output_dir, 'report.json')
        
        # Подсчитываем файлы
        graphs_count = len(os.listdir(os.path.join(self.output_dir, 'graphs'))) if os.path.exists(os.path.join(self.output_dir, 'graphs')) else 0
        tables_count = len(os.listdir(os.path.join(self.output_dir, 'tables'))) if os.path.exists(os.path.join(self.output_dir, 'tables')) else 0
        
        # Добавляем метаданные
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'generated_by': 'AI Metrics Collector v1.0',
            'summary': {
                'total_graphs_generated': graphs_count,
                'total_tables_generated': tables_count
            },
            'key_metrics': {
                'total_users': self.metrics.get('basic_stats', {}).get('total_users', 0),
                'total_materials': self.metrics.get('basic_stats', {}).get('total_materials', 0),
                'total_interactions': self.metrics.get('basic_stats', {}).get('total_interactions', 0),
                'avg_relevance_rate': self.metrics.get('recommendation_metrics', {}).get('avg_relevance_rate', 0),
                'model_active': self.metrics.get('recommendation_metrics', {}).get('model_active', False)
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ report.json - Итоговый отчет")
    
    def run(self):
        """Запускает полный сбор метрик и генерацию графиков"""
        print("\n" + "="*60)
        print("🤖 СБОРЩИК МЕТРИК ИИ-РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
        print("="*60)
        print(f"🕐 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.setup_output_dir()
            
            # Собираем все метрики
            self.collect_basic_stats()
            self.collect_user_activity_metrics()
            self.collect_material_popularity()
            self.collect_temporal_metrics()
            self.collect_recommendation_metrics()
            
            # Сохраняем и визуализируем
            self.save_tables()
            self.generate_all_graphs()
            self.generate_report()
            
            print("\n" + "="*60)
            print("✅ СБОР МЕТРИК ЗАВЕРШЕН УСПЕШНО!")
            print("="*60)
            print(f"📁 Все файлы сохранены в папке: {self.output_dir}")
            print(f"   - Таблицы: {self.output_dir}/tables/")
            print(f"   - Графики: {self.output_dir}/graphs/")
            print(f"   - Отчет: {self.output_dir}/report.json")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ ОШИБКА: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    collector = AIMetricsCollector()
    collector.run()
