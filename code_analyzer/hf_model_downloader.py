import os
from huggingface_hub import login, snapshot_download

class HFModelDownloader:
    """A utility class to download Hugging Face models."""
    _hf_token = None

    @staticmethod
    def get_local_model_path(model_id: str, local_model_dir: str = "./models") -> str:
        """Download and cache a Hugging Face model.
        Parameters
        ----------
        model_id : str
            The Hugging Face model ID (e.g., "google/gemma-3-1b-it").
        local_model_dir : str, optional
            The local directory to store the downloaded model. Defaults to "./models".
        """
        HFModelDownloader.__retrieve_hugging_face_token()
        HFModelDownloader.__authenticate()
        return HFModelDownloader.__download_model(model_id, local_model_dir)

    @staticmethod
    def __retrieve_hugging_face_token() -> None:
        """Retrieve Hugging Face token from env or prompt user."""
        if HFModelDownloader._hf_token is None:
            token = os.getenv("HF_TOKEN")
            if not token:
                token = input("Please enter your Hugging Face READ token: ")
                os.environ["HF_TOKEN"] = token  # Set for this session
            HFModelDownloader._hf_token = token

    @staticmethod
    def __authenticate() -> None:
        """Authenticate using the Hugging Face token."""
        login(token=HFModelDownloader._hf_token)

    @staticmethod
    def __download_model(model_id: str, local_model_dir: str) -> str:
        """Download and cache the model from Hugging Face if needed.
        Parameters
        ----------
        model_id : str
            The Hugging Face model ID (e.g., "google/gemma-3-1b-it").
        local_model_dir : str
            The local directory to store the downloaded model. Defaults to ./models/<model name>.
        """
        if local_model_dir == "./models":
            local_model_dir += f"/{model_id.replace('/', '--')}"

        local_model_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_model_dir,
            repo_type="model",
        )
        return local_model_path