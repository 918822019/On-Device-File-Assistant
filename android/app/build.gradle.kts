plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.example.filememoryagent"
    compileSdk = 34
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.example.filememoryagent"
        // minSdk 29（Android 10）：代码用到 MediaStore.Downloads、
        // MediaStore.VOLUME_EXTERNAL_PRIMARY 与 MediaStore.Files.FileColumns.RELATIVE_PATH，
        // 三者均为 API 29 引入。在 API 28 上它们会抛 NoClassDefFoundError /
        // NoSuchFieldError —— 且 registerWatchObservers() 在 onCreate 里没有
        // runCatching 包裹，RELATIVE_PATH 又出现在每次索引扫描的投影列里，
        // 属于必崩路径而非边缘情况。
        // DFlash 图需要的 Vulkan 1.1 入口自 API 28 起即具备，故提到 29 不影响它。
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk { abiFilters += "arm64-v8a" }

        buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:9000\"")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // Kotlin 2.0.0 起 Compose 编译器由 org.jetbrains.kotlin.plugin.compose 管理，
    // 不再需要（也不应设置）composeOptions.kotlinCompilerExtensionVersion。
    // 旧值 "1.5.15" 对应 Kotlin 1.9.x，与 2.0.0 并列会导致构建冲突。

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation(platform("androidx.compose:compose-bom:2024.09.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.foundation:foundation-layout")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.lifecycle:lifecycle-service:2.8.6")

    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.retrofit2:converter-gson:2.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
}
