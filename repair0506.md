* design a different prompt when cache hit base threshold so we let the matched code become a evidence to LLM, please design a prompt that LLM should first check the evidence is wether same as the user prompt ask, if so, use this as the result, if not, then start the following generation process.

* build a harness based on vllm tool calling, so that now the LLM can call for read/write files and check the fs to reach their needs. (not done yet)

* add a path that can save the result codes

* for future evaluation of each benchmark, build a script that can automatically run all the benchmark data in one dir, and gather the statistics during the process, including pass/ fail of syntax/function, or each tool usage, context length, iteration/retry amount, RAG usage etc.


* TODO: a website to make the user prompt more visible, and make all tags become a selectable options, so that we can easily test different settings and see the results. Also the result must display on the website with what the model generate in each generation process 
* TODO:now use all your budget to reconstruct the project to make the projhect more lightweight, more readable, and more flexiable to scale more module in the future.
* DONE: vectorDB build become parallelizable, so that we can build the vectorDB faster
