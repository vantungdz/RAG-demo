# RAG Web Demo

Minimal Retrieval Augmented Generation (RAG) web app: ask questions in the UI; the system retrieves relevant document chunks and the LLM answers only from that context.

## Setup

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Set API keys in `.env`**  
   Copy `.env.example` to `.env` and set:
   - **HUGGINGFACE_TOKEN** – for embeddings (free tier): https://huggingface.co/settings/tokens  
   - **OPENAI_API_KEY** – for Chat answers: https://platform.openai.com/api-keys

3. **Add documents**  
   Put `.txt` or `.md` files in the `data/` folder.

4. **Run ingestion** (chunk + embed → `embeddings.json`)
   ```bash
   npm run ingest
   ```

5. **Start the server**
   ```bash
   npm start
   ```
   Open http://localhost:3000 and submit a question.

## How the RAG pipeline works

1. **Ingestion (offline)**  
   Documents in `data/` are read, split into chunks of about 500–800 characters (with overlap to avoid cutting mid-sentence). Each chunk is sent to the **Hugging Face** Inference API (`sentence-transformers/all-MiniLM-L6-v2`). The resulting vectors and metadata are stored in `embeddings.json`. No database is used.

2. **Query (online)**  
   When the user submits a question, the backend embeds the question with the same embedding model. It then computes **cosine similarity** between the question vector and every stored chunk vector, and selects the **top 3** most similar chunks.

3. **Generation**  
   Those 3 chunks are concatenated and sent to the OpenAI Chat API as context, with a strict system instruction: answer **only** from this context. If the answer is not in the context, the model must respond with: *"The answer is not available in the provided documents."*

4. **Response**  
   The API reply is returned to the UI along with the retrieved chunks and their similarity scores, so you can see exactly what context was used for the answer.

The same Hugging Face embedding model is used for both ingestion and query so that similarity comparisons are meaningful. Chat answers still use the OpenAI API.
