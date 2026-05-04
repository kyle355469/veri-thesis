* for those failed case, don't save to the history cache, since they are not verified to be correct. Instead, we can log them separately for analysis and potential future improvements.
    * so only cache when pass 
    * else only log the failed attempt with relevant details (e.g., prompt, generated code, error messages) for debugging and analysis.

* cache threshold in pipeline.py need be further tuned.

* keywords version of the semantic cache establish & eval with direct version

* a script that can automatically prompt to let server maintain a high concurrency of requests.

* add a external module for actually verifying the code with testbenches.

* for the latest report, even cache doesn't hit, still output the highest ranked history example, it can help tune the hit threshold and also provide insights on the retrieval performance.

