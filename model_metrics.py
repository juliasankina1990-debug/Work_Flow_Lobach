#!/usr/bin/env python3
"""
Скрипт для сбора метрик производительности модели рекомендательной системы
Для курсовой работы - фокус на качестве работы ИИ
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Без отображения графиков
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter, defaultdict
import json
import traceback
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import db, User, Item, Interaction
from recommendation import recommender

# Настройка стилей графиков
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ModelMetricsCollector:
    def __init__(self):
        self.output_dir = 'model_metrics'
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
    
    def check_model_status(self):
        """Проверяет статус модели"""
        print("\n" + "="*60)
        print("🤖 СТАТУС МОДЕЛИ")
        print("="*60)
        
        with app.app_context():
            status = {
                'model_loaded': recommender.model is not None,
                'model_type': 'LightFM (гибридная коллаборативная)',
                'learning_rate': recommender.learning_rate,
                'epochs': recommender.epochs,
                'no_components': recommender.no_components,
                'users_in_dataset': len(recommender.user_id_map) if hasattr(recommender, 'user_id_map') else 0,
                'items_in_dataset': len(recommender.item_id_map) if hasattr(recommender, 'item_id_map') else 0,
                'user_features_count': len(recommender.user_features_list) if hasattr(recommender, 'user_features_list') else 0,
                'item_features_count': len(recommender.item_features_list) if hasattr(recommender, 'item_features_list') else 0,
            }
            
            print(f"✅ Модель загружена: {status['model_loaded']}")
            print(f"✅ Тип модели: {status['model_type']}")
            print(f"✅ Learning rate: {status['learning_rate']}")
            print(f"✅ Эпох обучения: {status['epochs']}")
            print(f"✅ Размерность: {status['no_components']}")
            print(f"✅ Пользователей в датасете: {status['users_in_dataset']}")
            print(f"✅ Материалов в датасете: {status['items_in_dataset']}")
            
            self.metrics['model_status'] = status
            return status
    
    def collect_interaction_statistics(self):
        """Собирает статистику по взаимодействиям для обучения"""
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА ОБУЧАЮЩИХ ДАННЫХ")
        print("="*60)
        
        with app.app_context():
            interactions = Interaction.query.all()
            
            # Группировка по типам
            action_types = {'view': 0, 'read': 0, 'save': 0, 'share': 0}
            for inter in interactions:
                action_types[inter.action] = action_types.get(inter.action, 0) + 1
            
            # Статистика по пользователям
            user_interactions = defaultdict(list)
            for inter in interactions:
                user_interactions[inter.user_id].append(inter)
            
            users_with_interactions = len(user_interactions)
            interactions_per_user = [len(inter) for inter in user_interactions.values()]
            
            # Статистика по материалам
            item_interactions = defaultdict(list)
            for inter in interactions:
                item_interactions[inter.item_id].append(inter)
            
            items_with_interactions = len(item_interactions)
            interactions_per_item = [len(inter) for inter in item_interactions.values()]
            
            # Веса взаимодействий
            weights = {'view': 1, 'read': 3, 'save': 5, 'share': 10}
            total_weight = sum(weights.get(inter.action, 1) for inter in interactions)
            
            stats = {
                'total_interactions': len(interactions),
                'action_distribution': action_types,
                'users_with_interactions': users_with_interactions,
                'avg_interactions_per_user': np.mean(interactions_per_user) if interactions_per_user else 0,
                'min_interactions_per_user': min(interactions_per_user) if interactions_per_user else 0,
                'max_interactions_per_user': max(interactions_per_user) if interactions_per_user else 0,
                'items_with_interactions': items_with_interactions,
                'avg_interactions_per_item': np.mean(interactions_per_item) if interactions_per_item else 0,
                'min_interactions_per_item': min(interactions_per_item) if interactions_per_item else 0,
                'max_interactions_per_item': max(interactions_per_item) if interactions_per_item else 0,
                'total_weighted_interactions': total_weight,
                'sparsity': 1 - (len(interactions) / (users_with_interactions * items_with_interactions)) if users_with_interactions * items_with_interactions > 0 else 1
            }
            
            print(f"✅ Всего взаимодействий: {stats['total_interactions']}")
            print(f"✅ Пользователей с активностью: {stats['users_with_interactions']}")
            print(f"✅ Материалов с взаимодействиями: {stats['items_with_interactions']}")
            print(f"✅ Среднее взаимодействий на пользователя: {stats['avg_interactions_per_user']:.2f}")
            print(f"✅ Разреженность матрицы: {stats['sparsity']:.4f}")
            
            print("\n📝 Распределение по типам:")
            for action, count in action_types.items():
                if count > 0:
                    print(f"   - {action}: {count} ({count/len(interactions)*100:.1f}%)")
            
            self.metrics['interaction_stats'] = stats
            self.metrics['interactions_per_user'] = interactions_per_user
            self.metrics['interactions_per_item'] = interactions_per_item
            return stats
    
    def evaluate_precision_at_k(self, k=5):
        """Оценивает Precision@k для модели"""
        print("\n" + "="*60)
        print(f"🎯 ОЦЕНКА PRECISION@{k}")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()
            precision_scores = []
            
            # Для каждого пользователя с активностью
            for user in users[:30]:  # Ограничим для скорости
                try:
                    # Получаем реальные прочтения пользователя (истинные значения)
                    user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
                    read_item_ids = set([i.item_id for i in user_reads])
                    
                    if len(read_item_ids) < 2:  # Нужно минимум 2 прочтения для оценки
                        continue
                    
                    # Разделяем на train/test (80/20)
                    read_list = list(read_item_ids)
                    random.shuffle(read_list)
                    split_idx = int(len(read_list) * 0.8)
                    train_reads = set(read_list[:split_idx])
                    test_reads = set(read_list[split_idx:])
                    
                    if not test_reads:
                        continue
                    
                    # Получаем рекомендации модели
                    recommendations = recommender.get_recommendations(user.id, n=k*2)
                    rec_item_ids = set([item.id for item in recommendations])
                    
                    # Считаем Precision@k
                    relevant_in_recs = len(rec_item_ids & test_reads)
                    precision = relevant_in_recs / min(k, len(rec_item_ids)) if rec_item_ids else 0
                    
                    precision_scores.append({
                        'user_id': user.id,
                        'username': user.username,
                        'train_size': len(train_reads),
                        'test_size': len(test_reads),
                        'relevant_in_recs': relevant_in_recs,
                        'precision': precision * 100
                    })
                    
                except Exception as e:
                    continue
            
            if precision_scores:
                avg_precision = np.mean([p['precision'] for p in precision_scores])
                print(f"✅ Средний Precision@{k}: {avg_precision:.2f}%")
                print(f"✅ Оценено пользователей: {len(precision_scores)}")
                
                # Топ результаты
                print("\n🏆 Лучшие результаты:")
                top_results = sorted(precision_scores, key=lambda x: x['precision'], reverse=True)[:3]
                for res in top_results:
                    print(f"   - {res['username']}: {res['precision']:.1f}% (train: {res['train_size']}, test: {res['test_size']})")
            else:
                avg_precision = 0
                print("⚠️ Недостаточно данных для оценки")
            
            metrics = {
                f'precision_at_{k}': round(avg_precision, 2),
                f'precision_at_{k}_details': precision_scores[:10]  # Первые 10 для примера
            }
            
            self.metrics.update(metrics)
            return metrics
    
    def evaluate_recall_at_k(self, k=5):
        """Оценивает Recall@k для модели"""
        print("\n" + "="*60)
        print(f"🎯 ОЦЕНКА RECALL@{k}")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()
            recall_scores = []
            
            for user in users[:30]:
                try:
                    user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
                    read_item_ids = set([i.item_id for i in user_reads])
                    
                    if len(read_item_ids) < 2:
                        continue
                    
                    read_list = list(read_item_ids)
                    random.shuffle(read_list)
                    split_idx = int(len(read_list) * 0.8)
                    train_reads = set(read_list[:split_idx])
                    test_reads = set(read_list[split_idx:])
                    
                    if not test_reads:
                        continue
                    
                    recommendations = recommender.get_recommendations(user.id, n=k*2)
                    rec_item_ids = set([item.id for item in recommendations])
                    
                    # Считаем Recall@k
                    relevant_in_recs = len(rec_item_ids & test_reads)
                    recall = relevant_in_recs / len(test_reads) if test_reads else 0
                    
                    recall_scores.append({
                        'user_id': user.id,
                        'username': user.username,
                        'test_size': len(test_reads),
                        'relevant_in_recs': relevant_in_recs,
                        'recall': recall * 100
                    })
                    
                except Exception as e:
                    continue
            
            if recall_scores:
                avg_recall = np.mean([r['recall'] for r in recall_scores])
                print(f"✅ Средний Recall@{k}: {avg_recall:.2f}%")
                print(f"✅ Оценено пользователей: {len(recall_scores)}")
            else:
                avg_recall = 0
                print("⚠️ Недостаточно данных для оценки")
            
            metrics = {
                f'recall_at_{k}': round(avg_recall, 2),
                f'recall_at_{k}_details': recall_scores[:10]
            }
            
            self.metrics.update(metrics)
            return metrics
    
    def calculate_ndcg(self, k=5):
        """Рассчитывает NDCG@k (Normalized Discounted Cumulative Gain)"""
        print("\n" + "="*60)
        print(f"🎯 ОЦЕНКА NDCG@{k}")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()
            ndcg_scores = []
            
            for user in users[:30]:
                try:
                    user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
                    read_item_ids = set([i.item_id for i in user_reads])
                    
                    if len(read_item_ids) < 2:
                        continue
                    
                    # Получаем все взаимодействия пользователя с весами
                    user_interactions = Interaction.query.filter_by(user_id=user.id).all()
                    item_weights = {}
                    for inter in user_interactions:
                        weight = {'view': 1, 'read': 3, 'save': 5, 'share': 10}.get(inter.action, 1)
                        item_weights[inter.item_id] = max(item_weights.get(inter.item_id, 0), weight)
                    
                    recommendations = recommender.get_recommendations(user.id, n=k)
                    
                    if not recommendations:
                        continue
                    
                    # Рассчитываем DCG и IDCG
                    dcg = 0
                    idcg = 0
                    
                    # Сортируем реальные веса по убыванию для идеального порядка
                    ideal_weights = sorted(item_weights.values(), reverse=True)[:k]
                    for i, weight in enumerate(ideal_weights):
                        idcg += weight / np.log2(i + 2)
                    
                    # Считаем DCG для рекомендаций
                    for i, item in enumerate(recommendations[:k]):
                        weight = item_weights.get(item.id, 0)
                        dcg += weight / np.log2(i + 2)
                    
                    ndcg = dcg / idcg if idcg > 0 else 0
                    
                    ndcg_scores.append({
                        'user_id': user.id,
                        'username': user.username,
                        'dcg': round(dcg, 4),
                        'idcg': round(idcg, 4),
                        'ndcg': round(ndcg * 100, 2)
                    })
                    
                except Exception as e:
                    continue
            
            if ndcg_scores:
                avg_ndcg = np.mean([n['ndcg'] for n in ndcg_scores])
                print(f"✅ Средний NDCG@{k}: {avg_ndcg:.2f}%")
                print(f"✅ Оценено пользователей: {len(ndcg_scores)}")
            else:
                avg_ndcg = 0
                print("⚠️ Недостаточно данных для оценки")
            
            metrics = {
                f'ndcg_at_{k}': round(avg_ndcg, 2),
                f'ndcg_at_{k}_details': ndcg_scores[:10]
            }
            
            self.metrics.update(metrics)
            return metrics
    
    def evaluate_coverage(self):
        """Оценивает покрытие модели (сколько материалов может рекомендовать)"""
        print("\n" + "="*60)
        print("🌍 ОЦЕНКА ПОКРЫТИЯ МОДЕЛИ")
        print("="*60)
        
        with app.app_context():
            all_items = Item.query.count()
            items_in_model = len(recommender.item_id_map) if hasattr(recommender, 'item_id_map') else 0
            
            # Проверяем, сколько разных материалов рекомендуют разным пользователям
            users = User.query.all()[:20]  # Ограничим
            recommended_items = set()
            
            for user in users:
                try:
                    recs = recommender.get_recommendations(user.id, n=10)
                    for item in recs:
                        recommended_items.add(item.id)
                except:
                    continue
            
            coverage = len(recommended_items) / items_in_model if items_in_model > 0 else 0
            coverage_total = len(recommended_items) / all_items if all_items > 0 else 0
            
            metrics = {
                'total_items_in_db': all_items,
                'items_in_model': items_in_model,
                'unique_items_recommended': len(recommended_items),
                'coverage_of_model': round(coverage * 100, 2),
                'coverage_of_total': round(coverage_total * 100, 2),
                'model_coverage_percent': f"{coverage*100:.1f}%"
            }
            
            print(f"✅ Всего материалов в БД: {metrics['total_items_in_db']}")
            print(f"✅ Материалов в модели: {metrics['items_in_model']}")
            print(f"✅ Уникальных материалов в рекомендациях: {metrics['unique_items_recommended']}")
            print(f"✅ Покрытие модели: {metrics['coverage_of_model']}%")
            print(f"✅ Покрытие от всех материалов: {metrics['coverage_of_total']}%")
            
            self.metrics['coverage'] = metrics
            return metrics
    
    def evaluate_diversity(self):
        """Оценивает разнообразие рекомендаций"""
        print("\n" + "="*60)
        print("🎨 ОЦЕНКА РАЗНООБРАЗИЯ")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()[:20]
            diversity_scores = []
            
            for user in users:
                try:
                    recs = recommender.get_recommendations(user.id, n=10)
                    
                    if len(recs) < 2:
                        continue
                    
                    # Считаем разнообразие по типам материалов
                    types = [item.type for item in recs if item.type]
                    unique_types = len(set(types))
                    type_diversity = unique_types / len(types) if types else 0
                    
                    # Считаем разнообразие по тегам
                    all_tags = []
                    for item in recs:
                        all_tags.extend(item.get_tags_list())
                    
                    unique_tags = len(set(all_tags))
                    tag_diversity = unique_tags / len(all_tags) if all_tags else 0
                    
                    diversity_scores.append({
                        'user_id': user.id,
                        'username': user.username,
                        'type_diversity': round(type_diversity * 100, 2),
                        'tag_diversity': round(tag_diversity * 100, 2),
                        'avg_diversity': round((type_diversity + tag_diversity) * 50, 2)
                    })
                    
                except:
                    continue
            
            if diversity_scores:
                avg_type_diversity = np.mean([d['type_diversity'] for d in diversity_scores])
                avg_tag_diversity = np.mean([d['tag_diversity'] for d in diversity_scores])
                avg_diversity = np.mean([d['avg_diversity'] for d in diversity_scores])
                
                print(f"✅ Среднее разнообразие по типам: {avg_type_diversity:.1f}%")
                print(f"✅ Среднее разнообразие по тегам: {avg_tag_diversity:.1f}%")
                print(f"✅ Общее разнообразие: {avg_diversity:.1f}%")
            else:
                avg_diversity = 0
                print("⚠️ Недостаточно данных")
            
            metrics = {
                'avg_type_diversity': round(avg_type_diversity, 2) if diversity_scores else 0,
                'avg_tag_diversity': round(avg_tag_diversity, 2) if diversity_scores else 0,
                'avg_diversity': round(avg_diversity, 2) if diversity_scores else 0,
                'diversity_details': diversity_scores[:10]
            }
            
            self.metrics['diversity'] = metrics
            return metrics
    
    def evaluate_novelty(self):
        """Оценивает новизну рекомендаций (не прочитанное ранее)"""
        print("\n" + "="*60)
        print("✨ ОЦЕНКА НОВИЗНЫ")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()[:20]
            novelty_scores = []
            
            for user in users:
                try:
                    # Что пользователь уже читал/смотрел
                    seen_items = set([i.item_id for i in Interaction.query.filter_by(user_id=user.id).all()])
                    
                    recs = recommender.get_recommendations(user.id, n=10)
                    
                    if not recs:
                        continue
                    
                    # Сколько новых (невиданных) материалов в рекомендациях
                    new_items = [item for item in recs if item.id not in seen_items]
                    novelty = len(new_items) / len(recs) * 100
                    
                    novelty_scores.append({
                        'user_id': user.id,
                        'username': user.username,
                        'seen_count': len(seen_items),
                        'new_in_recs': len(new_items),
                        'novelty': round(novelty, 2)
                    })
                    
                except:
                    continue
            
            if novelty_scores:
                avg_novelty = np.mean([n['novelty'] for n in novelty_scores])
                print(f"✅ Средняя новизна: {avg_novelty:.1f}% новых материалов")
                print(f"✅ Диапазон: {min(n['novelty'] for n in novelty_scores):.1f}% - {max(n['novelty'] for n in novelty_scores):.1f}%")
            else:
                avg_novelty = 0
                print("⚠️ Недостаточно данных")
            
            metrics = {
                'avg_novelty': round(avg_novelty, 2),
                'novelty_details': novelty_scores[:10]
            }
            
            self.metrics['novelty'] = metrics
            return metrics
    
    def analyze_explanations(self):
        """Анализирует качество объяснений"""
        print("\n" + "="*60)
        print("💡 АНАЛИЗ ОБЪЯСНЕНИЙ")
        print("="*60)
        
        with app.app_context():
            users = User.query.all()[:10]
            explanation_stats = []
            
            for user in users:
                try:
                    recs = recommender.get_recommendations(user.id, n=5)
                    
                    for item in recs:
                        explanation = recommender.explain_recommendation(user.id, item.id)
                        
                        # Классифицируем объяснение
                        exp_type = 'unknown'
                        if 'совпадают' in explanation.lower():
                            exp_type = 'feature_match'
                        elif 'отдел' in explanation.lower():
                            exp_type = 'department'
                        elif 'популярн' in explanation.lower():
                            exp_type = 'popularity'
                        elif 'предпочтений' in explanation.lower():
                            exp_type = 'preferences'
                        
                        explanation_stats.append({
                            'user_id': user.id,
                            'item_id': item.id,
                            'explanation': explanation,
                            'type': exp_type,
                            'length': len(explanation)
                        })
                        
                except:
                    continue
            
            # Статистика по типам
            type_counts = Counter([e['type'] for e in explanation_stats])
            
            metrics = {
                'total_explanations': len(explanation_stats),
                'explanation_types': dict(type_counts),
                'avg_explanation_length': np.mean([e['length'] for e in explanation_stats]) if explanation_stats else 0,
                'sample_explanations': explanation_stats[:5]
            }
            
            print(f"✅ Всего объяснений: {metrics['total_explanations']}")
            print(f"✅ Средняя длина: {metrics['avg_explanation_length']:.0f} символов")
            print("\n📝 Типы объяснений:")
            for exp_type, count in type_counts.items():
                print(f"   - {exp_type}: {count} ({count/len(explanation_stats)*100:.1f}%)")
            
            self.metrics['explanations'] = metrics
            return metrics
    
    def evaluate_at_different_k(self):
        """Оценивает метрики при разных k (1, 3, 5, 10)"""
        print("\n" + "="*60)
        print("📊 СРАВНЕНИЕ ПРИ РАЗНЫХ K")
        print("="*60)
        
        metrics_vs_k = []
        
        for k in [1, 3, 5, 10]:
            print(f"\n--- Оценка при k={k} ---")
            
            # Precision
            precisions = []
            recalls = []
            
            with app.app_context():
                users = User.query.all()[:20]
                
                for user in users:
                    try:
                        user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
                        read_item_ids = set([i.item_id for i in user_reads])
                        
                        if len(read_item_ids) < 2:
                            continue
                        
                        recommendations = recommender.get_recommendations(user.id, n=k)
                        rec_item_ids = set([item.id for item in recommendations])
                        
                        # Precision
                        relevant = len(rec_item_ids & read_item_ids)
                        precision = relevant / k if rec_item_ids else 0
                        precisions.append(precision)
                        
                        # Recall
                        recall = relevant / len(read_item_ids) if read_item_ids else 0
                        recalls.append(recall)
                        
                    except:
                        continue
            
            metrics_vs_k.append({
                'k': k,
                'precision': round(np.mean(precisions) * 100 if precisions else 0, 2),
                'recall': round(np.mean(recalls) * 100 if recalls else 0, 2),
                'f1': round(2 * (np.mean(precisions) * np.mean(recalls)) / (np.mean(precisions) + np.mean(recalls)) * 100 
                           if np.mean(precisions) + np.mean(recalls) > 0 else 0, 2)
            })
            
            print(f"   Precision@{k}: {metrics_vs_k[-1]['precision']}%")
            print(f"   Recall@{k}: {metrics_vs_k[-1]['recall']}%")
            print(f"   F1@{k}: {metrics_vs_k[-1]['f1']}%")
        
        self.metrics['metrics_vs_k'] = metrics_vs_k
        return metrics_vs_k
    
    def save_tables(self):
        """Сохраняет все данные в CSV таблицы"""
        print("\n" + "="*60)
        print("💾 СОХРАНЕНИЕ ТАБЛИЦ")
        print("="*60)
        
        tables_dir = os.path.join(self.output_dir, 'tables')
        
        # Сохраняем основные метрики
        metrics_summary = []
        for key, value in self.metrics.items():
            if isinstance(value, (int, float, str, bool)):
                metrics_summary.append({'metric': key, 'value': value})
        
        if metrics_summary:
            df_summary = pd.DataFrame(metrics_summary)
            df_summary.to_csv(os.path.join(tables_dir, '01_metrics_summary.csv'), index=False, encoding='utf-8')
            print("✅ 01_metrics_summary.csv - Сводка метрик")
        
        # Сохраняем метрики при разных k
        if 'metrics_vs_k' in self.metrics:
            df_vs_k = pd.DataFrame(self.metrics['metrics_vs_k'])
            df_vs_k.to_csv(os.path.join(tables_dir, '02_metrics_vs_k.csv'), index=False, encoding='utf-8')
            print("✅ 02_metrics_vs_k.csv - Метрики при разных k")
        
        # Сохраняем распределение объяснений
        if 'explanations' in self.metrics:
            df_exp = pd.DataFrame(self.metrics['explanations'].get('sample_explanations', []))
            if not df_exp.empty:
                df_exp.to_csv(os.path.join(tables_dir, '03_explanations.csv'), index=False, encoding='utf-8')
                print("✅ 03_explanations.csv - Примеры объяснений")
        
        # Сохраняем метрики разнообразия
        if 'diversity' in self.metrics and 'diversity_details' in self.metrics['diversity']:
            df_div = pd.DataFrame(self.metrics['diversity']['diversity_details'])
            if not df_div.empty:
                df_div.to_csv(os.path.join(tables_dir, '04_diversity.csv'), index=False, encoding='utf-8')
                print("✅ 04_diversity.csv - Разнообразие")
        
        # Сохраняем метрики новизны
        if 'novelty' in self.metrics and 'novelty_details' in self.metrics['novelty']:
            df_nov = pd.DataFrame(self.metrics['novelty']['novelty_details'])
            if not df_nov.empty:
                df_nov.to_csv(os.path.join(tables_dir, '05_novelty.csv'), index=False, encoding='utf-8')
                print("✅ 05_novelty.csv - Новизна")
    
    def generate_all_graphs(self):
        """Генерирует графики метрик модели"""
        print("\n" + "="*60)
        print("📈 ГЕНЕРАЦИЯ ГРАФИКОВ МОДЕЛИ")
        print("="*60)
        
        graphs_dir = os.path.join(self.output_dir, 'graphs')
        
        # График 1: Precision/Recall/F1 при разных k
        if 'metrics_vs_k' in self.metrics:
            plt.figure(figsize=(12, 6))
            
            df = pd.DataFrame(self.metrics['metrics_vs_k'])
            k_values = df['k'].values
            
            plt.plot(k_values, df['precision'], 'o-', linewidth=2, markersize=8, label='Precision', color='#3498db')
            plt.plot(k_values, df['recall'], 's-', linewidth=2, markersize=8, label='Recall', color='#2ecc71')
            plt.plot(k_values, df['f1'], '^-', linewidth=2, markersize=8, label='F1-score', color='#e74c3c')
            
            plt.xlabel('k (количество рекомендаций)', fontsize=12)
            plt.ylabel('Значение метрики (%)', fontsize=12)
            plt.title('Метрики качества при разном количестве рекомендаций', fontsize=16, pad=20)
            plt.xticks(k_values)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Добавляем значения
            for i, row in df.iterrows():
                plt.annotate(f"{row['precision']}%", (row['k'], row['precision']), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                plt.annotate(f"{row['recall']}%", (row['k'], row['recall']), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '01_metrics_vs_k.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 01_metrics_vs_k.png - Сравнение метрик при разных k")
        
        # График 2: Распределение типов объяснений
        if 'explanations' in self.metrics and 'explanation_types' in self.metrics['explanations']:
            plt.figure(figsize=(10, 8))
            
            exp_types = self.metrics['explanations']['explanation_types']
            labels = {
                'feature_match': 'Совпадение признаков',
                'department': 'По отделу',
                'popularity': 'По популярности',
                'preferences': 'Общие предпочтения',
                'unknown': 'Другие'
            }
            
            labels_ru = [labels.get(t, t) for t in exp_types.keys()]
            sizes = list(exp_types.values())
            
            if sizes and sum(sizes) > 0:
                colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
                plt.pie(sizes, labels=labels_ru, autopct='%1.1f%%',
                       colors=colors[:len(sizes)], startangle=90)
                plt.title('Типы объяснений рекомендаций', fontsize=16, pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '02_explanation_types.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 02_explanation_types.png - Распределение объяснений")
        
        # График 3: Распределение взаимодействий
        if 'interaction_stats' in self.metrics:
            plt.figure(figsize=(10, 6))
            
            actions = list(self.metrics['interaction_stats']['action_distribution'].keys())
            counts = list(self.metrics['interaction_stats']['action_distribution'].values())
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            
            bars = plt.bar(actions, counts, color=colors[:len(actions)])
            plt.xlabel('Тип взаимодействия', fontsize=12)
            plt.ylabel('Количество', fontsize=12)
            plt.title('Распределение типов взаимодействий в обучающих данных', fontsize=16, pad=20)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Добавляем значения
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '03_interaction_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 03_interaction_distribution.png - Распределение взаимодействий")
        
        # График 4: Гистограмма взаимодействий на пользователя
        if 'interactions_per_user' in self.metrics:
            plt.figure(figsize=(12, 6))
            
            plt.hist(self.metrics['interactions_per_user'], bins=20, color='#3498db', edgecolor='white', alpha=0.7)
            plt.axvline(np.mean(self.metrics['interactions_per_user']), color='red', linestyle='--', 
                       linewidth=2, label=f'Среднее: {np.mean(self.metrics["interactions_per_user"]):.1f}')
            plt.xlabel('Количество взаимодействий', fontsize=12)
            plt.ylabel('Количество пользователей', fontsize=12)
            plt.title('Распределение пользователей по активности', fontsize=16, pad=20)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '04_user_activity_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 04_user_activity_distribution.png - Активность пользователей")
        
        # График 5: Радар разнообразия (ИСПРАВЛЕННАЯ ВЕРСИЯ)
        if 'diversity' in self.metrics and 'coverage' in self.metrics and 'novelty' in self.metrics:
            plt.figure(figsize=(8, 8))
            
            categories = ['Типы', 'Теги', 'Новизна', 'Покрытие']
            
            # Получаем значения
            type_div = self.metrics['diversity'].get('avg_type_diversity', 0)
            tag_div = self.metrics['diversity'].get('avg_tag_diversity', 0)
            novelty = self.metrics.get('novelty', {}).get('avg_novelty', 0)
            coverage = self.metrics.get('coverage', {}).get('coverage_of_model', 0)
            
            # Для радара нужно одинаковое количество точек
            values = [type_div, tag_div, novelty, coverage]
            
            # Замыкаем круг - добавляем первое значение в конец
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]
            
            angles = np.linspace(0, 2 * np.pi, len(categories_closed), endpoint=False).tolist()
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values_closed, 'o-', linewidth=2, color='#9b59b6')
            ax.fill(angles, values_closed, alpha=0.25, color='#9b59b6')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title('Профиль качества модели', fontsize=16, pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, '05_model_quality_radar.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 05_model_quality_radar.png - Профиль качества")
        
        # График 6: Precision/Recall для отдельных пользователей
        if 'precision_at_5_details' in self.metrics:
            plt.figure(figsize=(12, 6))
            
            df_prec = pd.DataFrame(self.metrics['precision_at_5_details'][:15])
            
            if not df_prec.empty and 'precision' in df_prec.columns:
                x = range(len(df_prec))
                plt.bar(x, df_prec['precision'], color='#3498db')
                plt.xlabel('Пользователи', fontsize=12)
                plt.ylabel('Precision@5 (%)', fontsize=12)
                plt.title('Precision@5 для отдельных пользователей', fontsize=16, pad=20)
                
                # Используем username если есть, иначе индексы
                if 'username' in df_prec.columns:
                    plt.xticks(x, df_prec['username'].values, rotation=45, ha='right')
                else:
                    plt.xticks(x, [f"User {i+1}" for i in x])
                    
                plt.axhline(y=np.mean(df_prec['precision']), color='red', linestyle='--', 
                           label=f'Среднее: {np.mean(df_prec["precision"]):.1f}%')
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, '06_precision_per_user.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 06_precision_per_user.png - Precision по пользователям")
    
    def generate_report(self):
        """Генерирует итоговый отчет"""
        report_path = os.path.join(self.output_dir, 'model_report.json')
        
        graphs_count = len(os.listdir(os.path.join(self.output_dir, 'graphs'))) if os.path.exists(os.path.join(self.output_dir, 'graphs')) else 0
        tables_count = len(os.listdir(os.path.join(self.output_dir, 'tables'))) if os.path.exists(os.path.join(self.output_dir, 'tables')) else 0
        
        # Собираем ключевые метрики
        key_metrics = {
            'precision_at_1': self.metrics.get('precision_at_1', 0),
            'precision_at_3': self.metrics.get('precision_at_3', 0),
            'precision_at_5': self.metrics.get('precision_at_5', 0),
            'precision_at_10': self.metrics.get('precision_at_10', 0),
            'recall_at_5': self.metrics.get('recall_at_5', 0),
            'ndcg_at_5': self.metrics.get('ndcg_at_5', 0),
            'coverage': self.metrics.get('coverage', {}).get('coverage_of_model', 0),
            'diversity': self.metrics.get('diversity', {}).get('avg_diversity', 0),
            'novelty': self.metrics.get('novelty', {}).get('avg_novelty', 0),
        }
        
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': self.metrics.get('model_status', {}),
            'training_data': self.metrics.get('interaction_stats', {}),
            'key_metrics': key_metrics,
            'metrics_vs_k': self.metrics.get('metrics_vs_k', []),
            'summary': {
                'total_graphs': graphs_count,
                'total_tables': tables_count,
                'model_quality_score': np.mean([v for v in key_metrics.values() if isinstance(v, (int, float)) and v > 0])
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ model_report.json - Итоговый отчет о модели")
    
    def run(self):
        """Запускает полный сбор метрик модели"""
        print("\n" + "="*60)
        print("🤖 АНАЛИЗ МЕТРИК РЕКОМЕНДАТЕЛЬНОЙ МОДЕЛИ")
        print("="*60)
        print(f"🕐 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.setup_output_dir()
            
            # Собираем метрики модели
            self.check_model_status()
            self.collect_interaction_statistics()
            
            # Основные метрики качества
            self.evaluate_precision_at_k(k=1)
            self.evaluate_precision_at_k(k=3)
            self.evaluate_precision_at_k(k=5)
            self.evaluate_precision_at_k(k=10)
            
            self.evaluate_recall_at_k(k=5)
            self.calculate_ndcg(k=5)
            
            # Дополнительные метрики
            self.evaluate_coverage()
            self.evaluate_diversity()
            self.evaluate_novelty()
            self.analyze_explanations()
            
            # Сравнение при разных k
            self.evaluate_at_different_k()
            
            # Сохраняем и визуализируем
            self.save_tables()
            self.generate_all_graphs()
            self.generate_report()
            
            print("\n" + "="*60)
            print("✅ АНАЛИЗ МОДЕЛИ ЗАВЕРШЕН УСПЕШНО!")
            print("="*60)
            print(f"📁 Все файлы в папке: {self.output_dir}")
            print(f"   - Таблицы: {self.output_dir}/tables/")
            print(f"   - Графики: {self.output_dir}/graphs/")
            print(f"   - Отчет: {self.output_dir}/model_report.json")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ ОШИБКА: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    collector = ModelMetricsCollector()
    collector.run()
