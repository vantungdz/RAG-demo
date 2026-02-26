/**
 * RAG Web Demo server: serves UI and POST /api/query for retrieval + LLM answer.
 * Embeddings: Hugging Face (HUGGINGFACE_TOKEN). Chat: OpenAI (OPENAI_API_KEY).
 * Start with: npm start
 */
require('dotenv').config();
const path = require('path');
const fs = require('fs');
const express = require('express');
const OpenAI = require('openai');
const { cosineSimilarity, getEmbeddingHF } = require('./utils');

const app = express();
const EMBEDDINGS_PATH = path.join(__dirname, 'embeddings.json');
const CHAT_MODEL = 'gpt-4o-mini';
const TOP_K = 3;
const SIMILARITY_THRESHOLD = 0.75; // Lọc nhiễu: chỉ giữ chunk thực sự liên quan
const NOT_AVAILABLE_MSG = 'The answer is not available in the provided documents.';

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// RAG: Load embeddings once at startup (optional cache; we can reload per request for freshness)
function loadEmbeddings() {
  if (!fs.existsSync(EMBEDDINGS_PATH)) {
    throw new Error('embeddings.json not found. Run: npm run ingest');
  }
  return JSON.parse(fs.readFileSync(EMBEDDINGS_PATH, 'utf8'));
}

app.post('/api/query', async (req, res) => {
  try {
    const hfToken = process.env.HUGGINGFACE_TOKEN;
    if (!hfToken) {
      return res.status(500).json({ error: 'HUGGINGFACE_TOKEN is not set' });
    }

    const { question } = req.body;
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Missing or invalid "question" in body' });
    }

    // RAG step 1: Load stored embeddings
    const embeddings = loadEmbeddings();
    if (embeddings.length === 0) {
      return res.status(500).json({ error: 'No embeddings available. Run ingest with documents in data/.' });
    }

    // RAG step 2: Embed the user question (same HF model as ingest)
    const queryEmbedding = await getEmbeddingHF(question.trim(), hfToken);

    // RAG step 3: Cosine similarity, lọc theo threshold rồi lấy top-k
    const withScores = embeddings
      .map((entry) => ({
        ...entry,
        similarity: cosineSimilarity(queryEmbedding, entry.embedding),
      }))
      .sort((a, b) => b.similarity - a.similarity);
    const aboveThreshold = withScores.filter((c) => c.similarity > SIMILARITY_THRESHOLD);
    const topChunks = aboveThreshold.length > 0 ? aboveThreshold.slice(0, TOP_K) : withScores.slice(0, 1);

    // Build response payload (chunks always returned)
    const chunksPayload = topChunks.map(({ text, sourceFile, similarity }) => ({
      text,
      sourceFile,
      similarity: Math.round(similarity * 10000) / 10000,
    }));

    // RAG step 4 & 5: Try OpenAI Chat; on quota/error, return chunks + friendly message
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res.json({
        chunks: chunksPayload,
        answer: 'OpenAI API key is not set. Configure OPENAI_API_KEY to generate answers from the retrieved context.',
      });
    }

    const context = topChunks.map((c) => c.text).join('\n\n---\n\n');
    const systemPrompt = `Bạn CHỈ được sử dụng nội dung trong các đoạn sau đây để trả lời. Không sử dụng kiến thức bên ngoài.
Nếu thông tin không có trong các đoạn, bạn phải trả lời đúng câu sau (không thêm bớt): ${NOT_AVAILABLE_MSG}`;
    const userMessage = `Các đoạn tài liệu:\n\n${context}\n\nCâu hỏi: ${question}`;

    let answer;
    try {
      const chatResp = await openai.chat.completions.create({
        model: CHAT_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage },
        ],
        temperature: 0.2,
      });
      answer = chatResp.choices[0]?.message?.content?.trim() || NOT_AVAILABLE_MSG;
    } catch (chatErr) {
      const status = chatErr?.status || chatErr?.response?.status;
      const msg = chatErr?.message || String(chatErr);
      if (status === 429 || msg.includes('quota')) {
        answer = 'OpenAI API quota exceeded. Please check your plan and billing at platform.openai.com. In the meantime, use the retrieved chunks above to find the answer.';
      } else {
        answer = `Could not generate answer (${msg}). Please refer to the retrieved chunks above.`;
      }
    }

    res.json({ chunks: chunksPayload, answer });
  } catch (err) {
    console.error(err);
    const message = err.message || 'Server error';
    res.status(500).json({ error: message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`RAG demo server at http://localhost:${PORT}`);
});
