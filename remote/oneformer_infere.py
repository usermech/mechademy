import torch
import pickle
import gzip
import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from demo.defaults import DefaultPredictor

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

class OneFormerInference:
    SWIN_CFG_DICT = {
        "cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
        "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
        "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
    }
    DINAT_CFG_DICT = {
        "cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
        "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
        "ade20k": "configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml",
    }
    cpu_device = torch.device("cpu")

    def __init__(self, dataset, model_path, use_swin=False, device="cpu"):
        self.dataset = dataset
        self.model_path = model_path
        self.use_swin = use_swin
        self.predictor = None
        self.metadata = None
        self.setup_modules()
    
    def setup_cfg(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        
        cfg_path = self.SWIN_CFG_DICT[self.dataset] if self.use_swin else self.DINAT_CFG_DICT[self.dataset]
        cfg_path = os.path.join("/home/romer/umut/segmentation/OneFormer/", cfg_path)   # ADDED FOR LOCAL TESTING
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    def setup_modules(self):
        cfg = self.setup_cfg()
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
            from cityscapesscripts.helpers.labels import labels
            stuff_colors = [k.color for k in labels if k.trainId != 255]
            self.metadata = self.metadata.set(stuff_colors=stuff_colors)


    def infer_image(self, img, task="semantic"):
        """
        Perform inference on a single image.

        Args:
            img (numpy.ndarray): Input image array.
            task (str): Task type for OneFormer (e.g., 'semantic', 'instance', 'panoptic').
        Returns:
            Predictions for the specified task.
        """
        # Perform inference and extract panoptic segmentation and segments information
        predictions = self.predictor(img, task)
        panoptic_seg, segments_info = predictions["panoptic_seg"]

        # Move panoptic segmentation to CPU (if not already on CPU)
        panoptic_seg = panoptic_seg.to(self.cpu_device)

        # Clear GPU memory after operation
        torch.cuda.empty_cache()

        # Return both panoptic segmentation and segments information as a tuple
        return (panoptic_seg, segments_info)

    def infer_batch(self, image_dict, task="semantic"):
        """
        Perform inference on a batch of images.

        Args:
            image_dict (dict): Dict of images with keys as identifiers and values as image arrays.
            task (str): Task type for OneFormer (e.g., 'semantic', 'instance', 'panoptic').

        Returns:
            Dictinory of predictions for each image.
            he keys are the identifiers from the input dictionary, and the values are the predictions.
        """
        results = {}
        for key, img in image_dict.items():
            print(f"Processing image {key}...")
            predictions = self.infer_image(img, task=task)
            results[key] = predictions
        return results
        
def main():
    output_path = "/home/romer/umut/segmentation/OneFormer/outputs/predictions.pkl.gz"
    model_path = "/home/romer/umut/segmentation/OneFormer/250_16_dinat_l_oneformer_ade20k_160k.pth"
    oneformer = OneFormerInference(dataset="ade20k", model_path=model_path, use_swin=False)
    with gzip.open(f"/home/romer/umut/segmentation/OneFormer/rgb_observations.pkl.gz", "rb") as f:
        rgb_observations = pickle.load(f)
    print(f"Loaded {len(rgb_observations)} RGB observations.")
    predictions =  oneformer.infer_batch(rgb_observations, task="semantic")
    
    with gzip.open(output_path, "wb") as f:
        pickle.dump(predictions, f)

if __name__ == "__main__":
    main()
 