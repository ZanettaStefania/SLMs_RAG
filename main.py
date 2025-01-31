from imports import * 
from utils import *
from config import *

# Get process ID
pid = os.getpid()
#print(pid)

# Parse the model name input from command line
args = parse_args()
#print(args.test)
print(args.test is None)

qwen_rag_prompt_template = process_prompt(args)

model_id=inizialize(args)
print(model_id)

# Load the existing vector database instead of creating a new one
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY_DB, 
		  embedding_function=HuggingFaceBgeEmbeddings(model_name=DB_EMBD_MODEL_NAME, model_kwargs={"device": DEVICE_MAP}))

retriever = vectordb.as_retriever(search_type=SEARCH_TYPE_RETRIEVER, search_kwargs={"k": RETRIVED_K, "score_threshold": SIMILARITY_THRESHOLD})

#retriever = vectordb.as_retriever(search_kwargs={"k": RETRIVED_K})

model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=model, top_n=RETRIEVER_TOP_N)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor, base_retriever=retriever
)

#model_id = "/media/jetson/8822e6d5-68f8-44c2-8d88-adde671365d71/[download]Hug_model/Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)



pipe = pipeline(
          MODEL_REPLY,
          model=model,
          tokenizer=tokenizer,
          max_new_tokens=MAX_NEW_TOKENS,
          clean_up_tokenization_spaces=True,
          device_map=DEVICE_MAP
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Create the retrieval chain
combine_docs_chain = create_stuff_documents_chain(local_llm, PromptTemplate(input_variables=["context", "question"], template=qwen_rag_prompt_template))

# Create the full RAG chain with the compression retriever and the combine_docs_chain
rag_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)


if args.test is None:
	while True:
		q=input("How can I help you?\n>>")
		if q=="exit":
			break
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		context_, answer_ = split_string(llm_response['answer'])
		#print("Context:\n", context_)
		print("\nAnswer:\n", answer_)
		print("--------------------------------------------------------------------")
else:
	data = pd.read_csv(args.test)
	#print(len(data))
	for n in range(0, len(data)):
		q=data['Question'][n]
		g=data['ground_truths'][n]
		print("Question: \n", data['Question'][n])
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		#print("LLM_Response : ", llm_response['answer'])
		#process_llm_response(llm_response)
		context_, answer_ = split_string(llm_response['answer'])
		print("Time: ", (time.time() - start))
		process_question(model=args.model_name, question=n, answer=answer_, context=context_, time=(time.time() - start), data=data)
		print("Context:\n", context_)
		print("\nAnswer:\n", answer_)
		print("--------------------------------------------------------------------")
		#break


