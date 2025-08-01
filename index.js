//load the pdf

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"; //for chunking the pdf
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import * as dotenv from "dotenv";
dotenv.config();

async function indexDocument() {
  const PDF_PATH = "./dsa.pdf";
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs = await pdfLoader.load();


  // console.log(rawDocs.length);

  //chunking PDFLoader
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, // size of each chunk
    chunkOverlap: 200, // overlap between chunks
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

  // console.log(`Total chunks created: ${chunkedDocs.length}`);
  // // console.log(chunkedDocs[0].pageContent);

  // console.log(JSON.stringify(chunkedDocs.slice(0, 2), null, 2));

  //create embeddings vector
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });
  console.log("Embeddings created");

  //configure the database
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    //langchain (chunking, embedding, database)
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });

    console.log("Documents indexed successfully");


}

indexDocument();
