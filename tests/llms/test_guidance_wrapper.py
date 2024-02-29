from unittest.mock import patch

from openbb_chat.llms.guidance_wrapper import GuidanceWrapper


@patch("guidance.models.Transformers")
@patch("guidance.__call__")
def test_guidance_wrapper(mocked_transformers, mocked_call):
    guidance_wrapper = GuidanceWrapper()

    mocked_call.assert_called_once()
