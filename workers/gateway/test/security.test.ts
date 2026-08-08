import { describe, expect, it } from "vitest";
import { securityHeaders } from "../src/security";

describe("securityHeaders", () => {
  it("locks down framing and arbitrary connections", () => {
    const headers = securityHeaders();
    expect(headers.get("content-security-policy")).toContain("frame-ancestors 'none'");
    expect(headers.get("content-security-policy")).toContain("connect-src 'self'");
    expect(headers.get("x-content-type-options")).toBe("nosniff");
  });
});
