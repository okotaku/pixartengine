import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.datasets import (
    HFConditionDataset,
    HFConditionDatasetPreComputeEmbs,
)


class TestHFConditionDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFConditionDataset(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            csv="metadata_cn.csv")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
        assert isinstance(data["condition_img"], Image.Image)
        assert data["condition_img"].width == 400


class TestHFConditionDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFConditionDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            csv="metadata_cn.csv",
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
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400
        assert isinstance(data["condition_img"], Image.Image)
        assert data["condition_img"].width == 400
