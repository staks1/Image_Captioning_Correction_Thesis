# 🧠 Project Title: Industrial Captioning

This project demonstrates a 2 step pipeline to efficiently caption 
and also do retrieval for multimodal industrial datasets
The idea is to finetune specific layers of the 2 transformers of the CLIP multimodal model (https://github.com/openai/CLIP) so that we can teach the model specific domain knowledge. The results suggest that finetuning specific layers is possible when using appropriately sized datasets and augmentation can also be beneficial.


<p align="center">
  <img src="preview.png" width="700"/>
</p>


![System Architecture](./system.png)
---

### 📘 View the Notebook

[![View Notebook](https://img.shields.io/badge/View-Notebook-blue?logo=jupyter)](./demo-notebook.ipynb)

## 📦 Requirements

To run this demo project locally (optional), you'll need:

- Python 3.10+
- Git (for cloning the repo)
- Huggingface and accepting meta's (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) terms and conditions if you want to also run the caption correction part of the project

For Python-specific setup:

- All Python dependencies are listed in `requirements.txt`
- The Datasets and Models structure is created after cloning the project
and then running the python module `creating_datasets.py`.

---

## 🚀 How to Run the Demo (Optional)

> 💡 Skip this if you're just here to view the notebook — everything is pre-rendered.

The demo is available 
- In the demo we present:
    - caption retrieval
    - Image retrieval
    - Caption correction using LLAMA2 
    - We also calculate and save the embeddings
    - We calculate the ROUGE score for the demo dataset

If you'd like to run the demo project locally:

### 🐳 Running the notebook (study the whole pipeline)

1. Clone the repo
2. cd into training_and_evaluation
3. Run `creating_datasets.py`
4. After that the directory structure is created and you need to copy all files from `demo_dataset` into the `src/training_and_evaluation/Datasets/original/` directory
5. Create a python virtual environment 
6. Install the requirements
7.  Copy the `demo-notebook.ipynb` into `src/training_and_evaluation/`
8.  Run the notebook !

### Using Python (find the captions)
- A script is available for the demo-dataset to :
  - Calculate and save the embeddings
  - find the captions for each image
  - Calculate the metrics 

You can again run steps `1-6` and then run
the `inference_demo.py`. You then can inspect the embeddings and results
inside `src/training_and_evaluation/Models/zeroshot_clip/original/...`

### Using docker  (find the captions)
To run the captioning part you can also build the image and run the container
1. Follow steps `1-4`
2. Run the `run_docker_captioning.sh` which will build the image run the container and create the volumes `Models` where you can inspect the embeddings and results.


### Future steps
- [ ] Add unit testing
- [ ] Add function type hints
- [X] Containerize the demo

``````
