//! Binding generator entry point.
//!
//! UniFFI's CLI has to be built from the same crate so the generated Kotlin and
//! Swift match this crate's exact interface version. Run it via
//! `cargo run --bin uniffi-bindgen -- generate ...`.
fn main() {
    uniffi::uniffi_bindgen_main()
}
