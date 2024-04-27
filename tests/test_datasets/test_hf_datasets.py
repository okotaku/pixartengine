import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.datasets import HFDataset, HFDatasetPreComputeEmbs


class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDataset(
            dataset="tests/testdata/dataset", image_column="file_name")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a dog"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400

        dataset = HFDataset(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            csv="metadata2.csv")
        assert len(dataset) == 1

        data = dataset[0]
        assert data["text"] == "a cat"
        assert isinstance(data["img"], Image.Image)
        assert data["img"].width == 400


class TestHFDatasetPreComputeEmbs(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HFDatasetPreComputeEmbs(
            dataset="tests/testdata/dataset", image_column="file_name",
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
