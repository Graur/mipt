#!/bin/bash
set -e

# Создание пользователя test
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER test PASSWORD 'test_password';
    GRANT ALL PRIVILEGES ON DATABASE "$POSTGRES_USER" TO test;
EOSQL
