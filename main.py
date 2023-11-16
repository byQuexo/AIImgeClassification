from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image, ImageDraw
import requests

url = "https://imgs.search.brave.com/DkkwHvLKgHtrghG_ouzRwgBWl1cxvsAPu3ksuuVhmHs/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9tYWNo/aW5lbGVhcm5pbmdt/YXN0ZXJ5LmNvbS93/cC1jb250ZW50L3Vw/bG9hZHMvMjAxOS8w/MS9QbG90LW9mLWEt/U3Vic2V0LW9mLUlt/YWdlcy1mcm9tLXRo/ZS1DSUZBUi0xMC1E/YXRhc2V0LnBuZw"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.58)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
            # display the bounding box on the image
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, width=1, outline="red")
    draw.text((box[0], box[1]), model.config.id2label[label.item()], fill="red")
image.show()




