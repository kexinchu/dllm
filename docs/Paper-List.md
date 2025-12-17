### Title: ScaleLLM: A Resource-Frugal LLM Serving Framework by Optimizing End-to-End Efficiency
Institution: TensorOpera Inc   
Conference: ArXiv Sep 10 2024    
Paper Link: https://arxiv.org/html/2408.00008v1

##### Key Point
- Existing works only focus on optimizing individual subprocedures of LLM serving, but in commercial LLM applications, end-to-end latency, introduced from functionalities of the gateway, becomes the most significant bottleneck.
- Solutions:
    - 1, proposed a new router frameworks to reduce the gateway latency
    - 2, Load balance across multiple replicas of LLM service
        - Low concurrency (< 64 requests). Route requests to nodes with fewer replicas but higher tensor parallelism to optimize resource utilization for smaller batch computations.
        - High concurrency (≥ 64 requests). Route requests to nodes with more replicas but lower tensor parallelism, effectively distributing the workload to squeeze everything out of available compute by leveraging the power of replica parallelism.