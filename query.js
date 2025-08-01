import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

// Environment variable checks
if (!process.env.GEMINI_API_KEY) {
  console.error("Missing GEMINI_API_KEY in .env file.");
  process.exit(1);
}
if (!process.env.PINECONE_INDEX_NAME) {
  console.error("Missing PINECONE_INDEX_NAME in .env file.");
  process.exit(1);
}

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  // Add current question to history for rewrite context
  History.push({
    role: 'user',
    parts: [{ text: question }]
  });

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `
You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.
        `,
      },
    });
    return response.text;
  } catch (err) {
    console.error("Error rewriting query:", err.message);
    return question; // fallback to original
  } finally {
    History.pop(); // Remove rewrite prompt
  }
}

async function chatting(question) {
  try {
    // 1. Query rewriting for context independence
    const queries = await transformQuery(question);

    // 2. Vector embedding (embedding rewritten query may be more accurate)
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(queries);

    // 3. Pinecone search
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });
    const context = searchResults.matches
      .map(match => match.metadata.text)
      .join("\n\n---\n\n");

    // 4. Gemini answering
    History.push({
      role: 'user',
      parts: [{ text: queries }]
    });

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `
You have to behave like a Data Structure and Algorithm Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context: ${context}
        `,
      },
    });

    History.push({
      role: 'model',
      parts: [{ text: response.text }]
    });

    console.log("\n");
    console.log(response.text);
  } catch (err) {
    console.error("Error in chat:", err.message);
  }
}

async function main() {
  while (true) {
    const userProblem = readlineSync.question("Ask me anything--> ");
    await chatting(userProblem);
  }
}

main();
