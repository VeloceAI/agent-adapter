FROM swift:6.1
WORKDIR /app
COPY Package.swift .
RUN swift package resolve
COPY Sources Sources
RUN swift build -c release
CMD [".build/release/client"]
