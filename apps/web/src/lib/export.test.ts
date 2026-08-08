import { describe, expect, it } from "vitest";
import { demoFacts, POPULAR_COMPANIES } from "../data";
import { factsToCsv } from "./export";

describe("factsToCsv", () => {
  it("exports provenance with every fact", () => {
    const csv = factsToCsv(demoFacts(POPULAR_COMPANIES[0]!));
    expect(csv).toContain("Accession");
    expect(csv).toContain("SEC source");
    expect(csv.split("\r\n")).toHaveLength(6);
  });
});
