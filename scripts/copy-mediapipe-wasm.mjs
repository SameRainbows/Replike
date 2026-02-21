import { mkdir, readdir, copyFile, stat } from "node:fs/promises";
import path from "node:path";

async function pathExists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const repoRoot = process.cwd();

  const srcDir = path.join(
    repoRoot,
    "node_modules",
    "@mediapipe",
    "tasks-vision",
    "wasm"
  );
  const dstDir = path.join(repoRoot, "public", "mediapipe", "wasm");

  if (!(await pathExists(srcDir))) {
    throw new Error(`Source wasm directory not found: ${srcDir}`);
  }

  await mkdir(dstDir, { recursive: true });

  const entries = await readdir(srcDir);
  const wasmRelated = entries.filter((f) =>
    [".js", ".wasm", ".data"].some((ext) => f.endsWith(ext))
  );

  if (wasmRelated.length === 0) {
    throw new Error(`No wasm assets found in: ${srcDir}`);
  }

  for (const file of wasmRelated) {
    await copyFile(path.join(srcDir, file), path.join(dstDir, file));
  }

  process.stdout.write(
    `Copied ${wasmRelated.length} MediaPipe wasm assets to ${path.relative(
      repoRoot,
      dstDir
    )}\n`
  );
}

main().catch((err) => {
  process.stderr.write(`${err?.message ?? String(err)}\n`);
  process.exit(1);
});
