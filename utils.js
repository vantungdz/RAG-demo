const axios = require('axios');

// SỬA LẠI API_BASE VÀ CÁCH NỐI CHUỖI: Sử dụng hệ thống router mới của Hugging Face
const HF_API_BASE = 'https://router.huggingface.co/hf-inference/models';
const HF_EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';

// Đường dẫn URL hoàn chỉnh sẽ có dạng: 
// https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction
const HF_API_URL = `${HF_API_BASE}/${HF_EMBED_MODEL}/pipeline/feature-extraction`;

function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return dot / denom;
}

async function getEmbeddingHF(text, apiKey) {
  const res = await axios.post(
    HF_API_URL, 
    {
      inputs: text,
    },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      timeout: 60000,
      proxy: false, 
    }
  );

  // Trả về trực tiếp mảng embedding
  return res.data;
}

async function getEmbeddingsHF(texts, apiKey) {
  if (!Array.isArray(texts)) texts = [texts];

  const res = await axios.post(
    HF_API_URL,
    {
      inputs: texts,
    },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      timeout: 60000,
      proxy: false, 
    }
  );

  return res.data;
}

module.exports = { cosineSimilarity, getEmbeddingHF, getEmbeddingsHF };