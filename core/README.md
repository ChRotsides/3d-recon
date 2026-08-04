# Shared capture-guidance core

Plan §3.8 and §6.1, ADR-0001. The coverage voxel grid, next-best-station solver
and quality gates, in one Rust crate behind UniFFI so iOS and Android cannot
drift apart in capture behaviour.

## Test

```bash
cargo test          # 20 tests, no device needed
```

## Regenerate the bindings

The generated Kotlin and Swift are **committed**, so a client build needs no
Rust toolchain. Regenerate after changing the `ffi` module:

```bash
cargo build
cargo run --bin uniffi-bindgen -- generate \
    --library target/debug/librecon_core.so --language kotlin --out-dir bindings/kotlin
cargo run --bin uniffi-bindgen -- generate \
    --library target/debug/librecon_core.so --language swift  --out-dir bindings/swift

cp -r bindings/kotlin/uniffi/recon_core \
      ../clients/android/app/src/main/java/uniffi/
cp bindings/swift/recon_core.swift bindings/swift/recon_coreFFI.* \
      ../clients/ios/PoseLogger/Generated/
```

## Native libraries

The bindings compile without them; they are needed at *runtime*.

```bash
cargo install cargo-ndk
cargo ndk -t arm64-v8a -t armeabi-v7a -o ../clients/android/app/src/main/jniLibs build --release
# iOS: cargo build --release --target aarch64-apple-ios, then lipo into an xcframework
```

Not built here — the dev box has ~1 GB of disk free and each Rust target
toolchain is a couple of hundred megabytes. See docs/execution-status.md.

## The FFI layer is separate on purpose

`src/ffi.rs` wraps the core rather than annotating it. The core stays free of
FFI concerns and testable as plain Rust, and the clients get an API shaped for
what an app does — push a frame, get back what to show the user — instead of the
internals of a voxel grid.

`nag_message()` lives here too: capture guidance is a product surface, and if
each client wrote its own wording the two apps would say different things to the
same user. `test_client_core_parity.py` asserts neither client has its own copy.
