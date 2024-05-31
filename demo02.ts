import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import 'dotenv/config'
import { RunnableMap } from "@langchain/core/runnables";

// model
const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0
});

// ジョークを生成するChain
const jokeChain = PromptTemplate.fromTemplate("{topic}についてジョークを言ってください").pipe(model);

// ポエムを生成するChain
const poemChain = PromptTemplate.fromTemplate("{topic}についてポエムを言ってください").pipe(model);

// chain with pipe
const mapChain = RunnableMap.from({
  joke: jokeChain,
  poem: poemChain,
});


const result = await mapChain.invoke({ topic: "熊" });
console.log(result);
