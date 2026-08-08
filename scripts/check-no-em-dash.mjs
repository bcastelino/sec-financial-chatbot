import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const excludedDirectories = new Set([
  ".git",
  ".mypy_cache",
  ".playwright-mcp",
  ".pytest_cache",
  ".ruff_cache",
  ".venv",
  "__pycache__",
  "dist",
  "node_modules",
]);
const forbiddenCharacter = "\u2014";
const violations = [];

async function scan(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && excludedDirectories.has(entry.name)) continue;
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await scan(absolutePath);
      continue;
    }
    if (!entry.isFile()) continue;
    const contents = await readFile(absolutePath);
    if (contents.includes(0)) continue;
    const text = contents.toString("utf8");
    if (text.includes(forbiddenCharacter)) {
      violations.push(path.relative(root, absolutePath));
    }
  }
}

await scan(root);
if (violations.length > 0) {
  console.error(`Forbidden em dash found in:\n${violations.join("\n")}`);
  process.exitCode = 1;
} else {
  console.log("Punctuation check passed: no em dashes found.");
}
