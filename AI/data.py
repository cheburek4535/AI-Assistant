questions = [
    # Базовые вопросы о платформе
    "какого цвета небо",
    "что такое python",
    "как запустить fastapi",
    "почему база не работает",
    "где документация",

    # FastAPI и backend
    "как настроить postgresql в fastapi",
    "uvicorn не запускается ошибка",
    "как добавить cors в fastapi",
    "jwt токен как проверить",
    "sqlalchemy session закрывается",

    # B2B поставки/логистика
    "как создать поставку",
    "статусы заказа какие бывают",
    "счет-фактура где скачать",
    "доставка сколько стоит",
    "оплата по безналу как",

    # База данных
    "миграции alembic как запустить",
    "индексы на таблицу добавить",
    "запрос медленно работает",
    "foreign key не работает",
    "backup базы как сделать",

    # API endpoints
    "get поставки мои",
    "post заказ новый",
    "patch статус обновить",
    "delete товар из корзины",
    "auth Bearer токен как передать",

    # Пользователи/роли
    "роли admin manager client",
    "менеджер себя назначить",
    "пароль сбросить как",
    "2fa включить где",
    "логин не принимает",

    # Отчеты/аналитика
    "отчет по продажам за месяц",
    "экспорт excel поставки",
    "график остатков товаров",
    "топ поставщиков по объему",
    "средний чек заказов",

    # Интеграции
    "1с интеграция как настроить",
    "telegram бот уведомления",
    "email рассылка клиенты",
    "api ключи где взять",
    "webhook для заказов",

    # Проблемы/ошибки
    "502 bad gateway что делать",
    "docker контейнер не стартует",
    "память утечка python",
    "csrf token invalid",
    "timeout запроса увеличить"
]

answers = [
    # Базовые
    "небо синее днем",
    "python язык программирования",
    "uvicorn main:app --reload",
    "проверь postgresql подключение",
    "docs на сайте fastapi",

    # FastAPI и backend
    "DATABASE_URL в .env укажи postgresql://...",
    "pip install uvicorn[standard] && pip install psycopg2",
    "from fastapi.middleware.cors import CORSMiddleware",
    "@app.post('/login') def login(): return jwt.encode(...)",
    "use dependency async with db.get_session() as session:",

    # B2B поставки/логистика
    "POST /api/v1/supplies/ с данными поставки",
    "новый→в_работе→отгружен→оплачен→закрыт",
    "в личном кабинете → Мои счета → скачать PDF",
    "тарифы: 300р/кг или 25р/км рассчитать в /calc-delivery",
    "карта СБП или счет на оплату в /invoices",

    # База данных
    "alembic upgrade head из папки migrations",
    "CREATE INDEX idx_orders_date ON orders(created_at)",
    "EXPLAIN ANALYZE твой запрос + добавь индексы",
    "ON DELETE CASCADE в foreign key добавить",
    "pg_dump -U user dbname > backup_$(date +%Y%m%d).sql",

    # API endpoints
    "GET /api/v1/supplies/?client_id=me&status=open",
    "POST /api/v1/orders/ с JSON заказа",
    "PATCH /api/v1/orders/{id} {\"status\": \"shipped\"}",
    "DELETE /api/v1/cart/{product_id}",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",

    # Пользователи/роли
    "admin - все, manager - свои клиенты, client - свои заказы",
    "только admin может менять роли в /admin/users",
    "кнопка 'Забыли пароль?' → email с ссылкой",
    "в настройках профиля → Безопасность → 2FA",
    "проверь email или сбрось пароль",

    # Отчеты/аналитика
    "GET /reports/sales?month=12&year=2025 → CSV",
    "кнопка Excel в любом отчете или /export/orders",
    "Dashboard → Графики → Остатки по складам",
    "Reports → Топ-10 поставщиков по объему за год",
    "Analytics → Метрики → Средний чек 45 230₽",

    # Интеграции
    "POST /integrations/1c/webhook с XML заказов",
    "бот @B2BSupplyBot подключить в настройках",
    "/admin/newsletter → сегменты клиентов",
    "Settings → API Keys → Generate new token",
    "POST /webhooks/orders/created → JSON payload",

    # Проблемы/ошибки
    "nginx timeout увеличить или масштабировать backend",
    "docker-compose up --build или docker logs проверить",
    "psutil мониторинг добавить или gc.collect()",
    "csrf токен в cookies или headers передать",
    "TIMEOUT=300 в uvicorn или nginx proxy_read_timeout"
]

#print("Данные готовы:", len(questions), "примеров")


















