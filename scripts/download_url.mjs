// Parallel HTTP range downloader for large wheels (SJTU mirror etc.).
// usage: node download_url.mjs <url> <dest> [connections] [chunkMB]
import fs from "node:fs/promises";

const url = process.argv[2];
const dst = process.argv[3];
const CONC = Number(process.argv[4] || 8);
const CHUNK = Number(process.argv[5] || 8) * 1024 * 1024;
if (!url || !dst) {
  console.error("usage: node download_url.mjs <url> <dest> [connections] [chunkMB]");
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (n) => (n / 1e6).toFixed(0) + "MB";

let size = 0;
try {
  const head = await fetch(url, { method: "HEAD", signal: AbortSignal.timeout(60000) });
  if (head.ok) size = Number(head.headers.get("content-length") || 0);
} catch { /* fall through to range probe */ }
if (!size) {
  const r = await fetch(url, {
    headers: { Range: "bytes=0-0" },
    signal: AbortSignal.timeout(60000),
  });
  const cr = r.headers.get("content-range") || "";
  size = Number(cr.split("/").pop() || 0);
}
if (!size) throw new Error("no size (HEAD and range probe failed)");
console.log(`${dst}: ${fmt(size)}  conc=${CONC}`);

async function chunk(start, end, attempt = 1) {
  try {
    const r = await fetch(url, {
      headers: { Range: `bytes=${start}-${end}` },
      signal: AbortSignal.timeout(180000),
    });
    if (r.status !== 206) throw new Error(`status ${r.status}`);
    const b = Buffer.from(await r.arrayBuffer());
    if (b.length !== end - start + 1) throw new Error(`short ${b.length}`);
    return b;
  } catch (e) {
    if (attempt >= 6) throw e;
    await sleep(1000 * attempt);
    return chunk(start, end, attempt + 1);
  }
}

const fh = await fs.open(dst, "w");
await fh.truncate(size);
let next = 0, done = 0, last = Date.now(), lastDone = 0;
const t0 = Date.now();

async function worker() {
  for (;;) {
    const i = next++;
    const start = i * CHUNK;
    if (start >= size) return;
    const end = Math.min(size, start + CHUNK) - 1;
    const b = await chunk(start, end);
    await fh.write(b, 0, b.length, start);
    done += b.length;
    if (Date.now() - last > 5000) {
      const sp = (done - lastDone) / ((Date.now() - last) / 1000) / 1e6;
      console.log(`${fmt(done)}/${fmt(size)}  ${sp.toFixed(2)} MB/s`);
      last = Date.now();
      lastDone = done;
    }
  }
}

try {
  await Promise.all(Array.from({ length: CONC }, worker));
} finally {
  await fh.close();
}
const st = await fs.stat(dst);
if (st.size !== size) throw new Error(`size ${st.size} != ${size}`);
console.log(`DONE ${fmt(size)} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);