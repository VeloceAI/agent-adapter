FROM dart:stable
WORKDIR /app
COPY pubspec.yaml .
RUN dart pub get
COPY bin bin
CMD ["dart", "run", "bin/client.dart"]
