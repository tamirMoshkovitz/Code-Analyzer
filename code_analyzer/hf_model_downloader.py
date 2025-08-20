import os
import sys
from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.hf_api import RepositoryNotFoundError

AUTHORIZATION_TRIALS_NUMBER: int = 4


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
        for i in range(AUTHORIZATION_TRIALS_NUMBER):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=local_model_dir,
                    repo_type="model",
                )
                print(f"Model {model_id} fetched from {local_model_path}")
                return local_model_path
            except (HfHubHTTPError, RepositoryNotFoundError) as e:
                if isinstance(e, HfHubHTTPError):
                    code = getattr(getattr(e, "response", None), "status_code", None)
                    if code not in {401, 403}:  # Not an authorization error
                        raise
                if i == AUTHORIZATION_TRIALS_NUMBER - 1:
                    print(f"Failed to fetch model {model_id} after {AUTHORIZATION_TRIALS_NUMBER} attempts.",
                          file=sys.stderr)
                    return None
                HFModelDownloader.__retrieve_hugging_face_token()
                HFModelDownloader.__authenticate()
