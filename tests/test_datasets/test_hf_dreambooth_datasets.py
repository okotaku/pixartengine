import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.datasets import (
    HFDreamBoothDataset,
    HFDreamBoothDatasetPreComputeEmbs,
)


class TestHFDreamBoothDataset(RunnerTestCase):

    def test_dataset(self):
        dataset = HFDreamBoothDataset(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 5

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 1815

    def test_dataset_from_local(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset_db",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

    def test_dataset_from_local_with_csv(self):
        dataset = HFDreamBoothDataset(
            dataset="tests/testdata/dataset",
            csv="metadata.csv",
            image_column="file_name",
            instance_prompt="a photo of sks dog")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a photo of sks dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400


class TestHFDreamBoothDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="diffusers/dog-example",
            instance_prompt="a photo of sks dog",
            model="PixArt-alpha/PixArt-XL-2-1024-MS",
            tokenizer=dict(type=AutoTokenizer.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            text_encoder=dict(type=T5EncoderModel.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            device="cpu")
        assert len(dataset) == 5

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (120, 32)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 1815

    def test_dataset_from_local(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset_db",
            instance_prompt="a photo of sks dog",
            model="PixArt-alpha/PixArt-XL-2-1024-MS",
            tokenizer=dict(type=AutoTokenizer.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            text_encoder=dict(type=T5EncoderModel.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            device="cpu")
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (120, 32)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

    def test_dataset_from_local_with_csv(self):
        dataset = HFDreamBoothDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset",
            csv="metadata.csv",
            image_column="file_name",
            instance_prompt="a photo of sks dog",
            model="PixArt-alpha/PixArt-XL-2-1024-MS",
            tokenizer=dict(type=AutoTokenizer.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            text_encoder=dict(type=T5EncoderModel.from_pretrained,
                        pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
            device="cpu")
        assert len(dataset) == 1

        data = dataset[0]
        assert "text" not in data
        assert np.array(data["prompt_embeds"]).shape == (120, 32)
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
