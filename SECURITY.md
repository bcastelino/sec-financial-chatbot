# Security policy

Please report credential exposure, arbitrary-fetch paths, citation spoofing,
quota bypasses, prompt-injection issues, or cross-origin vulnerabilities
privately through GitHub's security advisory feature. Do not open a public issue
for an exploitable vulnerability.

The browser must never receive OpenRouter, Cloudflare, R2, or SEC identity
credentials. Production requests are same-origin, Turnstile-protected where
they incur cost, size-limited, and proxied through the Worker gateway.

Filing content is treated as untrusted data. Model output may cite only source
IDs issued by the server for the current retrieval set.
