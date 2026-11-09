FROM maven:3-eclipse-temurin-21
WORKDIR /app
COPY pom.xml .
RUN mvn -q -B dependency:go-offline
COPY src src
RUN mvn -q -B -o compile
CMD ["mvn", "-q", "-B", "-o", "exec:java"]
