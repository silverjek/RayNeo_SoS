plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.activeperception"
    compileSdk = 36
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.sos.rayneox3.final"
        minSdk = 29
        targetSdk = 36
        versionCode = 2
        versionName = "1.1-final-safe"

        ndk {
            // RayNeo X3 Pro is arm64; excluding emulator ABIs cuts the deployable APK size.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17")
            }
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }


    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // Keep native TFLite libraries extracted for predictable loading on the Android 12
    // RayNeo vendor image.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // On-device YOLOv8n. TFLite + Adreno GPU delegate is the working accelerator path on
    // S25 (Android 15+, Hexagon V79): ORT 1.22's QNN EP fails device-create on V79; TFLite
    // GPU delegate goes through OpenGL/Vulkan and is documented + maintained.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    // Cloud-offload HTTP client (OffloadClient) -- async POST of frames to the GPU server.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
