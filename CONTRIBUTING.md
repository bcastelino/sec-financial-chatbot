# Contributing

Filing Room welcomes focused issues and pull requests.

1. Create a branch from `main`.
2. Keep SEC and provider access behind the gateway/API boundary.
3. Add a fixture for extraction or financial-selection behavior changes.
4. Add UI tests for user-visible state changes.
5. Run the root and API quality checks documented in the README.

Do not commit filing downloads, API credentials, `.env` files, generated
clients, or Cloudflare account identifiers. Test fixtures must be small,
attributed excerpts that are necessary to exercise parser behavior.

Commit messages should use a concise conventional prefix such as `feat:`,
`fix:`, `test:`, `docs:`, or `chore:`.
