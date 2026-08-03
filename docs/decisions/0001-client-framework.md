# ADR-0001 — Native shells with a shared Rust core

**Status:** accepted · **Date:** 2026-08-03
**Resolves:** plan §11 open decision 1, which had to be settled before Phase 1.

## Context

Plan §3.8 puts the choice as: native shells (Swift/ARKit + Kotlin/ARCore) over a
shared Rust core via UniFFI, versus Unity + AR Foundation 6.x. Its own guidance:
*"Pick native if this is a product; pick Unity if you need a demo in three
weeks."*

## Decision

**Native shells with a shared Rust core.**

## Why

1. **The interesting math is not UI.** The coverage voxel grid, the
   next-best-station solver and the quality gates (plan §6.1) are the
   differentiating feature. In Rust they are unit-testable on a desktop with no
   device and no emulator — 20 tests run in 0.18 s — and they run unmodified on
   a Jetson for ground-truth work. Three of those tests caught real
   specification bugs (see execution-status §2.1–2.3) that would otherwise have
   surfaced as "the app never says the room is finished" during a field trial.
2. **One implementation, not two.** Any behavioural difference between iOS and
   Android capture shows up in Phase 0 as a per-platform quality difference
   nobody can explain. A shared core makes drift impossible rather than
   merely discouraged.
3. **Custom CV is the whole product.** Plan §3.6 puts DA3-Small on-device via
   ExecuTorch on the guidance path. Native gives direct access to `ARFrame`'s
   captured pixel buffer and ARCore's `Image` planes; in Unity that path runs
   through AR Foundation's abstraction with extra copies, on the hot path of a
   1–2 Hz inference budget.
4. **Coordinate frames.** Plan §7 lists flipped scenes as a failure mode and
   attributes it to Apple-vs-Google handedness. The precise version: ARKit and
   ARCore are *both* right-handed, Y-up, −Z forward. The left-handed frame
   people actually hit is **Unity's**. Choosing native means we never enter it.

## Consequences

- Two UI codebases. Accepted: the pose loggers are ~600 lines each and the
  production capture UI is thin over the core.
- UniFFI bindings are needed before Phase 3. Until then the two clients keep
  their keyframe rules hand-synchronised, which is called out in both files as a
  stopgap.
- Rust toolchain in CI, plus `cargo-ndk` and an iOS static-lib build.
- The viewer stays an embedded WebView with Babylon.js 8+ (plan §3.8), so no
  native splat renderer is written twice.

## Revisit if

Both clients end up as thin shells with negligible native code *and* the Rust
core stops growing — at which point the two-codebase cost stops buying anything.
