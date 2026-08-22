To install dependencies, run:
```sh
uv sync
```

To add a new dependency, run:
```sh
uv add <dependency>
```

To run a single chunking file, use:
```sh
uv run --env-file .env.local -m chunking.chunk_agentic
```
replacing `chunk_agentic` with whatever file you want to run.

To submit documents to pageindex, use:
```sh
uv run --env-file .env.local page_index/pageindex_submission.py
```

To run the entire chunking evaluation, use:
```sh
uv run --env-file .env.local chunking_evaluation.py
```