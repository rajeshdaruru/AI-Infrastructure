# AI-Infrastructure
A working record of my path into ML platform/infra engineering — GPU and distributed training fundamentals, model serving and inference, and the AI infra tooling around them, documented as I build it.

## GPU Training Fundamentals

- Model Training and Use Trained Model: [sentiment-fine-tune-notebook.ipynb](sentiment-fine-tune-notebook.ipynb) ([Kaggle](https://www.kaggle.com/code/rajeshdaruru/1-financial-sentiment-fine-tune-notebook)) — fine-tunes a Hugging Face `transformers` model for financial sentiment classification on the Financial PhraseBank dataset, saves the trained model in portable Hugging Face format, and reloads the saved model to run predictions.
- Train on a Rented GPU: [cifar10-gpu-training/](cifar10-gpu-training/) — rents a cloud GPU, trains ResNet-18 on CIFAR-10 with standard single-GPU speed-ups, and verifies the saved checkpoint reloads correctly.

## License

Code is under the [MIT License](LICENSE). Datasets keep their own licenses, noted in each notebook.
