FROM ruby:3.4-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential >/dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Gemfile .
RUN bundle install --quiet
COPY client.rb .
CMD ["bundle", "exec", "ruby", "client.rb"]
