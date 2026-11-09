FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release && rm -r src
COPY src src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
COPY --from=build /src/target/release/client /client
CMD ["/client"]
