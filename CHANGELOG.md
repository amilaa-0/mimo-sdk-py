# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-04

### Added
- Synchronous `MiMo` and asynchronous `AsyncMiMo` clients with shared
  configuration and retry policy.
- `chat.completions.create` and `chat.completions.stream` covering both
  request shapes (single response and Server-Sent Events).
- `speech.create` for the `mimo-v2.5-tts` text-to-speech model, with
  automatic base64 decoding into `SpeechResult.audio_bytes`.
- `vision.describe` and `build_user_message` helpers around the
  `mimo-v2-omni` multimodal model. Accepts local paths, raw bytes, or
  remote URLs.
- Built-in catalog (`mimo.list_models`, `mimo.get_model`) of MiMo V2/V2.5
  models with capability metadata.
- Typed exception hierarchy mapping HTTP status codes to subclasses of
  `MiMoError`. Honors `Retry-After` for rate limits.
- `mimo` command-line tool (`pip install 'mimo-sdk[cli]'`) with `chat`,
  `speak`, `vision`, and `models` subcommands.
- 43 unit tests using `respx` to mock the transport, covering streaming,
  retries, error mapping, and request shaping.
