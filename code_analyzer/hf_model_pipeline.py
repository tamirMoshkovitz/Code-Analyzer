from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines import SUPPORTED_TASKS
from typing import Literal

PipelineTasks = Literal[
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

    def __init__(self,
                 local_model_path: str,
                 *,
                 task: PipelineTasks = "text-generation",
                 do_sample: bool = True,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 padding: bool = True,
                 truncation: bool = True,
                 ) -> None:
        """Initialize the HFModelPipeline.
        Parameters
        ----------
        local_model_path : str
            The local path to the Hugging Face model directory.
        task : PipelineTasks, optional
            The task to perform with the model. Default is "text-generation".
        do_sample : bool, optional
            Whether to enable sampling. Defaults to True.
        temperature : float, optional
            The temperature for sampling. Defaults to 0.7.
        top_p : float, optional
            The top-p value for nucleus sampling. Defaults to 0.9.
        top_k : float, optional
            The top-k value for sampling. Defaults to 40.
        padding : bool, optional
            Whether to pad the input sequences. Defaults to True.
        truncation : bool, optional
            Whether to truncate the input sequences. Defaults to True.
        Raises
        ------
        ValueError
            If the given `task` is not supported by Hugging Face pipelines.
        """
        self._model = None
        self._tokenizer = None
        self._generator = None
        self._local_model_path = local_model_path

        self.__load_tokenizer()
        self.__load_model()
        self.__create_pipeline(task, do_sample, temperature, top_p, top_k, padding, truncation)

    def get_model_response(self,
                           prompt: str,
                           *,
                           return_full_text: bool = False,
                           max_new_tokens: int = 8192,
                           **gen_overrides,
                           ) -> str:
        """Generate a response from the model for a given prompt.
        Parameters
        ----------
        prompt : str
            The input text prompt.
        return_full_text : bool, optional
            Whether to return the full text (prompt + completion) from the pipeline. Defaults to False.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate for this call. Defaults to 8192 (gemma3 1b it default max_length).
        gen_overrides : dict, optional
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

    def __load_tokenizer(self) -> None:
        """Load the tokenizer from the local path."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._local_model_path, model_max_length=8192)

    def __load_model(self) -> None:
        """Load the model from the local path."""
        self._model = AutoModelForCausalLM.from_pretrained(self._local_model_path)

    def __create_pipeline(self,
                          task,
                          do_sample,
                          temperature,
                          top_p,
                          top_k,
                          padding,
                          truncation
                          ) -> None:
        """Creates a text-generation pipeline
        Parameters
        ----------
        task : PipelineTasks
            The task to perform with the model.
        do_sample : bool
            Whether to enable sampling.
        temperature : float
            The temperature for sampling.
        top_p : float
            The top-p value for nucleus sampling.
        top_k : float
            The top-k value for sampling.
        padding : bool
            Whether to pad the input sequences.
        truncation : bool
            Whether to truncate the input sequences.
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
            padding=padding,
            truncation=truncation,
        )
