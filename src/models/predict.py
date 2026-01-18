import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

# FIXME: change hardcoded paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG
from face_parsing.model import BiSeNet

def post_process_mask(mask_np):
    if mask_np.max() == 0: return mask_np
    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_smooth = np.ones((3, 3), np.uint8)
    final_mask = cv2.erode(opened, kernel_smooth, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel_smooth, iterations=1)
    return final_mask

class HairSegmenter:
    def __init__(self, device):
        self.device = device
        self.model_path = Path(CONFIG['segmentation']['bisenet_model_path'])
        self.hair_class_id = CONFIG['segmentation']['hair_class_id']
        
        if not self.model_path.is_absolute():
            self.model_path = project_root / self.model_path

        if not self.model_path.exists():
             raise FileNotFoundError(f"BiSeNet weights not found at: {self.model_path}")

        print(f"Loading BiSeNet from: {self.model_path}")
        self.net = BiSeNet(n_classes=19)
        # weights_only=False for Pytorch 2.6+
        self.net.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=False))
        self.net.to(device)
        self.net.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def remove_background(self, image_path):
        img_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        inp_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_all = self.net(inp_tensor)
            out = out_all[0] if isinstance(out_all, tuple) else out_all
            out_up = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            probabilities = F.softmax(out_up, dim=1)
            mask_prob = probabilities.squeeze(0)[self.hair_class_id].cpu().numpy()
            mask_binary = (mask_prob > 0.5).astype(np.uint8) * 255
            mask_final = post_process_mask(mask_binary)

        img_np = np.array(img_pil)
        mask_3d = (mask_final > 0).astype(np.uint8)[:, :, np.newaxis]
        masked_img_np = img_np * mask_3d
        return Image.fromarray(masked_img_np)

# classifier system

class HairClassifierSystem:
    def __init__(self, run_id, device):
        self.device = device
        self.run_dir = project_root / "checkpoints" / run_id
        
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Training run folder not found: {self.run_dir}")

        print(f"Loading model hierarchy from: {run_id}")

        self.structure = {
            'h0': {'folder': 'h0_main',     'suffix': 'H0_Main'},
            'h1': {'folder': 'h1_straight', 'suffix': 'H1_Subclass'},
            'h2': {'folder': 'h2_wavy',     'suffix': 'H2_Subclass'},
            'h3': {'folder': 'h3_curly',    'suffix': 'H3_Subclass'}
        }
        self.labels_map = {0: ["1a", "1b", "1c"], 1: ["2a", "2b", "2c"], 2: ["3a", "3b", "3c"]}
        
        self.models = {}
        for key in ['h0', 'h1', 'h2', 'h3']:
            self.models[key] = self._load_single_model(key)

    def _extract_arch_from_name(self, filename, suffix):
        stem = Path(filename).stem
        if stem.startswith("best_model_"): stem = stem.replace("best_model_", "")
        suffix_part = f"_{suffix}"
        if stem.endswith(suffix_part): return stem.replace(suffix_part, "")
        return stem.split('_')[0]

    def _build_model_skeleton(self, arch_name, num_classes=3):
        if not hasattr(models, arch_name): raise ValueError(f"Unknown architecture: {arch_name}")
        model_fn = getattr(models, arch_name)
        try: model = model_fn(weights=None)
        except TypeError: model = model_fn(pretrained=False)
        
        if hasattr(model, 'fc'): 
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
             if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
             else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        return model

    def _fix_state_dict_keys(self, state_dict):
        # fc.1 -> fc (for models with dropout)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.1.'):
                new_key = k.replace('fc.1.', 'fc.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def _load_single_model(self, key):
        info = self.structure[key]
        folder_path = self.run_dir / info['folder']
        suffix = info['suffix']
        candidates = list(folder_path.glob(f"*_{suffix}.pth"))
        if not candidates: raise FileNotFoundError(f"Model file with suffix {suffix} not found in {folder_path}")
        
        filepath = candidates[0]
        arch = self._extract_arch_from_name(filepath.name, suffix)
        print(f"   - [{key.upper()}] Loading {arch} from file {filepath.name}")
        
        model = self._build_model_skeleton(arch, num_classes=3)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        raw_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        fixed_state_dict = self._fix_state_dict_keys(raw_state_dict)
        
        try:
            model.load_state_dict(fixed_state_dict)
        except RuntimeError:
            model.load_state_dict(fixed_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            # main model (H0)
            out0 = self.models['h0'](img_tensor)
            probs0 = torch.softmax(out0, dim=1)
            main_idx = torch.argmax(probs0).item()
            main_conf = probs0[0, main_idx].item()

            if main_idx == 0: sub_model = self.models['h1']
            elif main_idx == 1: sub_model = self.models['h2']
            elif main_idx == 2: sub_model = self.models['h3']
            else: return None, None

            # submodels (H1, H2, H3)
            out_sub = sub_model(img_tensor)
            probs_sub = torch.softmax(out_sub, dim=1)
            
            # get all probabilities for subclasses
            all_probs_list = probs_sub[0].tolist()
            sub_idx = torch.argmax(probs_sub).item()
            
            sub_class_names = self.labels_map[main_idx]
            sub_class_details = {}
            for i, prob in enumerate(all_probs_list):
                name = sub_class_names[i]
                sub_class_details[name] = prob

            label = self.labels_map[main_idx][sub_idx]
            
            return label, {
                "main_idx": main_idx, 
                "main_conf": main_conf, 
                "sub_idx": sub_idx, 
                "all_sub_probs": sub_class_details
            }

# helpers

def find_latest_run_id():
    checkpoints_dir = project_root / "checkpoints"
    if not checkpoints_dir.exists(): raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")
    folders = [f for f in checkpoints_dir.iterdir() if f.is_dir()]
    valid_folders = [f for f in folders if f.name[0].isdigit()]
    if not valid_folders: raise FileNotFoundError("No model folders found in checkpoints/")
    latest_folder = sorted(valid_folders, key=lambda x: x.name, reverse=True)[0]
    return latest_folder.name

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG['dataset']['target_size_h'], CONFIG['dataset']['target_size_w'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG['augmentation']['imagenet_mean'], 
                             std=CONFIG['augmentation']['imagenet_std'])
    ])

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    try:
        run_id = args.run_id
        if run_id is None:
            print("No --run_id provided. Searching for the latest model...")
            run_id = find_latest_run_id()
            print(f"Found latest run: {run_id}")
        
        segmenter = HairSegmenter(device)
        classifier = HairClassifierSystem(run_id, device)
        
        print(f"Processing: {Path(args.image).name}")
        masked_image_pil = segmenter.remove_background(args.image)
        
        transform = get_inference_transforms()
        input_tensor = transform(masked_image_pil).unsqueeze(0)

        print("Classifying...")
        label, details = classifier.predict(input_tensor)

        print("\n" +"-" * 40)
        print(f"FINAL RESULT: {label}")
        print("-" * 40)
        print(f"   Main Type: {details['main_idx']+1} (Confidence: {details['main_conf']:.2%})")
        print("-" * 40)
        print("   Subclass Details:")
        
        # sort and display all subclasses
        sorted_sub = sorted(details['all_sub_probs'].items(), key=lambda x: x[1], reverse=True)
        
        for name, prob in sorted_sub:
            is_winner = "*" if name == label else " "
            print(f"   {is_winner} {name}: {prob:.2%}")
            
        print("-" * 40 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()