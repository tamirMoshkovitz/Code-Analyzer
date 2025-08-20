import os
from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.hf_api import RepositoryNotFoundError


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
        return HFModelDownloader.__download_model(model_id, local_model_dir)

    @staticmethod
    def __retrieve_hugging_face_token() -> None:
        """Retrieve Hugging Face token from env or prompt user.
        Raises
        -------
        EnvironmentError
            If no Hugging Face token is found in the environment variables.
        """
        if HFModelDownloader._hf_token is None:
            token = os.getenv("HF_TOKEN")
            HFModelDownloader._hf_token = token

    @staticmethod
    def __authenticate() -> None:
        """Authenticate using the Hugging Face token."""
        if HFModelDownloader._hf_token:
            login(token=HFModelDownloader._hf_token)

    @staticmethod
    def __download_model(model_id: str, local_model_dir: str) -> str | None:
        """Download and cache the model from Hugging Face if needed.
        Parameters
        ----------
        model_id : str
            The Hugging Face model ID (e.g., "google/gemma-3-1b-it").
        local_model_dir : str
            The local directory to store the downloaded model. Defaults to ./models/<model name>.
        Returns
        -------
        str or None
            The local path to the downloaded model, or None if the download fails.
        """
        if local_model_dir == "./models":
            local_model_dir += f"/{model_id.replace('/', '--')}"

        print(f"Attempting to fetch model {model_id} with no token")  # notify user that no token is used for the first attempt
        try:
            local_model_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_model_dir,
                repo_type="model",
            )
        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            print(f"Failed to fetch model {model_id}: {e} with no token, trying with authentication")
            if isinstance(e, HfHubHTTPError):
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code not in {401, 403}:  # Not an authorization error
                    raise
            HFModelDownloader.__retrieve_hugging_face_token()
            if not HFModelDownloader._hf_token:
                raise EnvironmentError("No Environment variable 'HF_TOKEN' found.")
            HFModelDownloader.__authenticate()
        # try again with authentication
            local_model_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_model_dir,
                repo_type="model",
            )
        print(f"Model {model_id} fetched from {local_model_path}")
        return local_model_path
