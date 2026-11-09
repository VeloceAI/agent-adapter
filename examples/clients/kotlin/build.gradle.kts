plugins {
    kotlin("jvm") version "2.4.21"
    application
}

repositories { mavenCentral() }

dependencies {
    implementation("io.modelcontextprotocol:kotlin-sdk:0.16.0")
    implementation("io.ktor:ktor-client-cio:3.5.2")
}

kotlin { jvmToolchain(21) }

application { mainClass.set("ClientKt") }
