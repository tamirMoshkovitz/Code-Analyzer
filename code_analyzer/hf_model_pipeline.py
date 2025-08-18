import os
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines import SUPPORTED_TASKS
from typing import Literal

PipelineTasks = Literal[ # For IDE warnings
    "feature-extraction",
    "sentiment-analysis",
    "ner",
    "question-answering",
    "fill-mask",
    "summarization",
    "translation",
    "text2text-generation",
    "text-generation",
    "zero-shot-classification",
    "conversational",
]

class HFModelPipeline:
    """A wrapper for Hugging Face model pipeline to generate text responses."""

    def __init__(self, model_id: str = "google/gemma-3-1b-it",
                 local_model_dir: str = "gemma-3-1b-it-model") -> None:
        """Initialize the HFModelPipeline, including downloading the model to a local directory.
        Args:
            model_id (str): The Hugging Face model ID to use.
            local_model_dir (str): The local directory to cache the model.
        """
        self._model = None
        self._tokenizer = None
        self._generator = None
        self._model_id = model_id
        self._local_model_dir = local_model_dir
        self._local_model_path = None

        self.__retrieve_hugging_face_token()
        self.__authenticate()
        self.__download_model()
        self.__load_tokenizer()
        self.__load_model()
        self.__create_pipeline()

    def get_model_response(self, prompt: str, *, return_full_text: bool = False,
                           max_new_tokens: int = 500, **gen_overrides) -> str:
        """Generate a response from the model for a given prompt.
        Parameters
        ----------
        prompt : str
            The input text prompt.
        return_full_text : bool, optional
            Whether to return the full text (prompt + completion) from the pipeline. Defaults to False.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate for this call. Defaults to 500.
        gen_overrides : dict
            Additional generation parameters to override the stored defaults for this call (e.g., temperature=0.8).
        Returns
        -------
        str
            The generated response text.
        """
        # Merge stored defaults with per-call overrides
        params = dict(self._gen_kwargs)
        params.update(gen_overrides)

        outputs = self._generator(
            prompt,
            return_full_text=return_full_text,
            max_new_tokens=max_new_tokens,
            **params,
        )
        return outputs[0]["generated_text"]

    def __retrieve_hugging_face_token(self):
        """Retrieve Hugging Face token from os or prompt user."""
        self._hf_token = os.getenv('HF_TOKEN')
        # If not set, ask the user to enter the token
        if not self._hf_token:
            self._hf_token = input("Please enter your Hugging Face READ token: ")
            os.environ['HF_TOKEN'] = self._hf_token  # Set the token for the current session

    def __authenticate(self) -> None:
        """Authenticate using the Hugging Face token."""
        login(token=self._hf_token)

    def __download_model(self) -> None:
        """Download and cache the model from Hugging Face if needed."""
        self._local_model_path = snapshot_download(
            self._model_id,
            local_dir=self._local_model_dir,
            repo_type="model",
        )

    def __load_tokenizer(self) -> None:
        """Load the tokenizer from the local path."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._local_model_path, model_max_length=8192)

    def __load_model(self) -> None:
        """Load the model from the local path."""
        self._model = AutoModelForCausalLM.from_pretrained(
            self._local_model_path
        )

    def __create_pipeline(self,
                          task: PipelineTasks = "text-generation",
                          do_sample: bool = True,
                          temperature: float = .7,
                          top_p: float = .9,
                          top_k: float = 40,
                          max_length: int = 8192,
                          padding: bool = True,
                          truncation: bool = True
                          ) -> None:
        """Creates a text-generation pipeline
        ----------
        task : PipelineTasks, optional
            The task to perform with the model. Defaults to "text-generation".
        do_sample : bool, optional
            Whether to enable sampling. Defaults to True.
        temperature : float, optional
            The temperature for sampling. Defaults to 0.7.
        top_p : float, optional
            The top-p value for nucleus sampling. Defaults to 0.9.
        top_k : float, optional
            The top-k value for sampling. Defaults to 40.
        max_length : int, optional
            The maximum length of the generated text. Defaults to 8192.
        padding : bool, optional
            Whether to pad the input sequences. Defaults to True.
        truncation : bool, optional
            Whether to truncate the input sequences. Defaults to True.
        Raises
        ------
        ValueError
            If the given `task` is not supported by Hugging Face pipelines.
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Invalid task '{task}'. Supported tasks: {list(SUPPORTED_TASKS.keys())}")

        self._generator = pipeline(
            task,
            model=self._model,
            tokenizer=self._tokenizer,
        )
        # Set generation parameters
        self._gen_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )
