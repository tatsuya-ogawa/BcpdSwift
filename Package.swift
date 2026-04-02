// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BCPD",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "BCPD",
            targets: ["BCPD"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "BCPD",
            dependencies: [],
            path: "Sources/BCPD"
        ),
        .testTarget(
            name: "BCPDTests",
            dependencies: ["BCPD"],
            path: "Tests/BCPDTests",
            resources: [
                .process("Resources")
            ]
        ),
    ]
)
