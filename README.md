# Mastering RAG: Techniques for Improved Retrieval and Generation

Code that accompanies the talk "Mastering RAG: Techniques for Improved Retrieval and Generation" first
held at
[Bergisches Entwicklerforum](https://www.meetup.com/bergisches-entwicklerforum/events/300781375).
This repository showcases steps involved in building a RAG pipeline for a QA system on the
[python scheduler library by Digon.IO](https://github.com/digonio/scheduler).

The components covered for the RAG pipeline are build upon the
[haystack](https://github.com/deepset-ai/haystack) framework and
[ollama](https://github.com/ollama/ollama) backend. By default the 4bit quantized version of
gemma2:2b is used, therefore the hardware requirements are minimal and no external API keys
are needed.

## Structure

The implementation of the individual components can be found in `src/talk_rag_scheduler`.
Most of the files can be run as a script and offer command line arguments for configuration.
Get information about the available run-time parameters with:

```bash
python -m talk_rag_scheduler.indexing_pipeline --help
python -m talk_rag_scheduler.retrieval_pipeline --help
python -m talk_rag_scheduler.rag_pipeline --help
python -m talk_rag_scheduler.rag_pipeline_concurrent --help
python -m talk_rag_scheduler.bench_retrieval --help
```

The individual steps come with additional jupyter notebooks.
Start with [01_start_here.ipynb](notebooks/00_start_here.ipynb).

## Installation

This project has only been tested with Python 3.12, it might work with 3.11 as well.

```console
pip install -r requirements.txt
pip install -e .
```

* Make sure you have ollama running and listening as specified in `const.py`
* You can get pretty formatted markdown output in the terminal with
  [rich-cli](https://github.com/Textualize/rich-cli)
  `pip install rich-cli`.

  Try it out:

  ```bash
  python -m talk_rag_scheduler.rag_pipeline "What are the features of the scheduler?"  | rich - --markdown
  ```

## Topics for future work

RAG system

* Multi-turn chat (possibly using haystack's ChatMessage object and streaming callback)
* Document preprocessing
* persistent databases & indexes

Coding

* Use environment variables for configuration instead of hardcoded `const.py`
