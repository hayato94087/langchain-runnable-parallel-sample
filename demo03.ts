import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "dotenv/config";

// model
const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
});

// vector store
const vectorstore = await MemoryVectorStore.fromDocuments(
  [
    { pageContent: "山田バクテリアは細胞の力源です", metadata: {} },
    { pageContent: "山田バクテリアは複数のセルから構成されます", metadata: {} },
    { pageContent: "吉田バクテリアは細胞の力源です", metadata: {} },
  ],
  new OpenAIEmbeddings()
);
const retriever = vectorstore.asRetriever();

// prompt template 
const template = `文脈に基づいて質問に答えてください:
{context}

質問: {question}`;
const prompt = PromptTemplate.fromTemplate(template);

// chain with pipe
const formatDocs = (docs: Document[]) => docs.map((doc) => doc.pageContent);
const retrievalChain = RunnableSequence.from([
  { context: retriever.pipe(formatDocs), question: new RunnablePassthrough() },
  prompt,
  model,
  new StringOutputParser(),
]);

// invoke
const result = await retrievalChain.invoke(
  "山田バクテリアとは何ですか？"
);
console.log(result);
