/**
 * RAG Ingestion: read .txt/.md from data/, chunk, embed with Hugging Face, write embeddings.json.
 * Run with: npm run ingest
 * Requires HUGGINGFACE_TOKEN in .env (or environment).
 */
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { getEmbeddingsHF } = require('./utils');

const DATA_DIR = path.join(__dirname, 'data');
const EMBEDDINGS_PATH = path.join(__dirname, 'embeddings.json');

// Chunk nhỏ hơn (200–300 ký tự) để mỗi mục/đoạn là 1 chunk; overlap nhỏ tránh cắt giữa câu.
const CHUNK_SIZE = 300;
const CHUNK_OVERLAP = 50;

// Tách theo section: số thứ tự (1. 2. 3.) hoặc "Mục N", hoặc ## markdown
const SECTION_SPLIT = /\n(?=\d+\.\s|Mục\s*\d+|\s*##\s)/i;

// HF free API has limits; small batches to avoid rate limit
const BATCH_SIZE = 5;

/**
 * Chunk theo đoạn nhỏ (200–300 ký tự). Ưu tiên tách theo section (1. / Mục N / ##), không có thì tách theo kích thước.
 */
function chunkText(text) {
  const sections = text.split(SECTION_SPLIT).map((s) => s.trim()).filter((s) => s.length > 0);
  const list = sections.length >= 2 ? sections : [text.trim()];

  const chunks = [];
  for (const section of list) {
    if (section.length <= CHUNK_SIZE) {
      if (section.length > 0) chunks.push(section);
      continue;
    }
    let start = 0;
    const len = section.length;
    while (start < len) {
      let end = Math.min(start + CHUNK_SIZE, len);
      if (end < len) {
        const segment = section.slice(start, end + 1);
        const nn = segment.lastIndexOf('\n\n');
        const dot = segment.lastIndexOf('. ');
        const sp = segment.lastIndexOf(' ');
        if (nn >= 60) end = start + nn + 2;
        else if (dot >= 60) end = start + dot + 1;
        else if (sp >= 60) end = start + sp + 1;
      }
      const chunk = section.slice(start, end).trim();
      if (chunk.length > 0) chunks.push(chunk);
      start = end - CHUNK_OVERLAP;
      if (start <= 0 || start >= len) start = end;
      if (start >= len) break;
    }
  }
  return chunks;
}

async function main() {
  const apiKey = process.env.HUGGINGFACE_TOKEN;
  if (!apiKey) {
    console.error('Missing HUGGINGFACE_TOKEN. Set it in .env or environment.');
    process.exit(1);
  }

  // 1. Read all .txt and .md files from data/
  const files = fs.readdirSync(DATA_DIR).filter((f) => {
    const ext = path.extname(f).toLowerCase();
    return ext === '.txt' || ext === '.md';
  });

  if (files.length === 0) {
    console.log('No .txt or .md files in data/. Add documents and run again.');
    process.exit(0);
  }

  const allEntries = [];
  let globalId = 0;

  for (const file of files) {
    const filePath = path.join(DATA_DIR, file);
    const content = fs.readFileSync(filePath, 'utf8');
    const chunks = chunkText(content);

    for (const text of chunks) {
      allEntries.push({
        id: String(globalId++),
        text,
        sourceFile: file,
        embedding: null,
      });
    }
  }

  console.log(`Chunked ${allEntries.length} chunks from ${files.length} file(s). Embedding...`);

  // 2. Generate embeddings via Hugging Face API (small batches for free tier)
  for (let i = 0; i < allEntries.length; i += BATCH_SIZE) {
    const batch = allEntries.slice(i, i + BATCH_SIZE);
    const texts = batch.map((e) => e.text);
    const vectors = await getEmbeddingsHF(texts, apiKey);
    for (let j = 0; j < batch.length; j++) {
      batch[j].embedding = vectors[j];
    }
    console.log(`Embedded ${Math.min(i + BATCH_SIZE, allEntries.length)} / ${allEntries.length}`);
  }

  // 3. Write embeddings.json (array of { id, text, embedding, sourceFile })
  const output = allEntries.map(({ id, text, embedding, sourceFile }) => ({
    id,
    text,
    embedding,
    sourceFile,
  }));
  fs.writeFileSync(EMBEDDINGS_PATH, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Wrote ${EMBEDDINGS_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
