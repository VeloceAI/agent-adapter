FROM golang:1.25 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go .
RUN CGO_ENABLED=0 go build -o /client .

FROM gcr.io/distroless/static-debian12
COPY --from=build /client /client
ENTRYPOINT ["/client"]
