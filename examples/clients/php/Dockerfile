FROM php:8.4-cli
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends unzip >/dev/null && rm -rf /var/lib/apt/lists/*
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer
WORKDIR /app
COPY composer.json .
RUN composer install --no-dev --no-interaction --quiet
COPY client.php .
CMD ["php", "client.php"]
