FROM gradle:9.8.2-jdk21
WORKDIR /app
COPY settings.gradle.kts build.gradle.kts ./
RUN gradle --no-daemon -q dependencies > /dev/null
COPY src src
RUN gradle --no-daemon -q installDist
CMD ["build/install/client/bin/client"]
