// Специфичные функции для демо-панели
document.addEventListener('DOMContentLoaded', function() {
    console.log('Demo panel loaded');
    
    // Обработка выбора пользователя
    const userSelect = document.getElementById('userSelect');
    if (userSelect) {
        userSelect.addEventListener('change', function() {
            const userId = this.value;
            // Здесь можно загружать данные для выбранного пользователя
            console.log('Selected user for analysis:', userId);
            
            // Обновляем текст в колонках в зависимости от пользователя
            updateAlgorithmDescriptions(userId);
        });
    }
    
    // Обработка формы добавления материала
    const addForm = document.getElementById('addMaterialForm');
    if (addForm) {
        addForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const material = {
                title: formData.get('title'),
                type: formData.get('type'),
                department: formData.get('department'),
                tags: formData.get('tags').split(',').map(t => t.trim()),
                content: formData.get('content')
            };
            
            console.log('New material:', material);
            
            // В реальном проекте здесь был бы POST запрос на сервер
            alert('Материал добавлен (демо-режим)');
            
            // Очищаем форму
            this.reset();
        });
    }
    
    // Обработка кнопок редактирования/удаления
    document.querySelectorAll('.btn-warning, .btn-danger').forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.classList.contains('btn-warning') ? 'редактирование' : 'удаление';
            const materialTitle = this.closest('li').firstChild.textContent.trim();
            alert(`${action} материала "${materialTitle}" (демо-режим)`);
        });
    });
});

// Функция обновления описаний алгоритмов в зависимости от пользователя
function updateAlgorithmDescriptions(userId) {
    const descriptions = {
        // Для разработчика
        '1': {
            content: 'Учтены ваши теги: #python, #бэкенд, #архитектура',
            collaborative: 'Учтена активность отдела "Разработка"',
            hybrid: 'Комбинация #python, #бэкенд и активности отдела'
        },
        // Для HR
        '2': {
            content: 'Учтены ваши теги: #hr, #онбординг, #рекрутинг',
            collaborative: 'Учтена активность отдела "Управление персоналом"',
            hybrid: 'Комбинация #hr, #онбординг и активности отдела'
        },
        // Для аналитика
        '3': {
            content: 'Учтены ваши теги: #аналитика, #дашборды, #sql',
            collaborative: 'Учтена активность отдела "Аналитика"',
            hybrid: 'Комбинация #аналитика, #sql и активности отдела'
        }
    };
    
    const desc = descriptions[userId] || descriptions['1'];
    
    // Обновляем текст в колонках
    const contentCol = document.querySelector('.col-md-4:first-child small.text-muted');
    const collabCol = document.querySelector('.col-md-4:nth-child(2) small.text-muted');
    const hybridCol = document.querySelector('.col-md-4:last-child small.text-muted');
    
    if (contentCol) contentCol.textContent = 'Контентная: ' + desc.content;
    if (collabCol) collabCol.textContent = 'Коллаборативная: ' + desc.collaborative;
    if (hybridCol) hybridCol.textContent = 'Гибридная: ' + desc.hybrid;
}
