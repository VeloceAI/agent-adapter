// swift-tools-version:6.1
import PackageDescription

let package = Package(
    name: "client",
    dependencies: [.package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", exact: "0.12.1")],
    targets: [
        .executableTarget(name: "client", dependencies: [.product(name: "MCP", package: "swift-sdk")])
    ]
)
