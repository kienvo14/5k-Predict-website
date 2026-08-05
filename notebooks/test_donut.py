"""
Quick pretrained-model probe: can an off-the-shelf model read the mileage
numbers out of a Strava chart screenshot without any fine-tuning?

Tries two candidates per image:
  1. Donut (naver-clova-ix/donut-base-finetuned-cord-v2) - document VQA/parsing model
  2. Tesseract OCR via pytesseract - plain text OCR

No training happens here, only inference. Runs fine on CPU (slow-ish for Donut).

Requires the Tesseract binary to be installed separately and on PATH
(pip's pytesseract package is only a wrapper around it):
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - macOS:   brew install tesseract
  - Linux:   apt install tesseract-ocr
"""

import glob
import os

from PIL import Image

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
IMAGE_GLOBS = ["*.jpg", "*.jpeg", "*.png"]


def find_images():
    paths = []
    for pattern in IMAGE_GLOBS:
        paths.extend(glob.glob(os.path.join(SCREENSHOTS_DIR, pattern)))
    return sorted(paths)


def load_donut():
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def run_donut(processor, model, image):
    import torch

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    return sequence.strip()


def run_ocr(image):
    import pytesseract

    # winget/UB-Mannheim installer's default location; pytesseract otherwise
    # relies on `tesseract` being on PATH.
    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

    text = pytesseract.image_to_string(image)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def main():
    images = find_images()
    if not images:
        print(f"No images found in {SCREENSHOTS_DIR}")
        return

    print(f"Found {len(images)} image(s) in {SCREENSHOTS_DIR}\n")

    print("Loading Donut model (naver-clova-ix/donut-base-finetuned-cord-v2)...")
    try:
        processor, model = load_donut()
        donut_available = True
    except Exception as e:
        print(f"Could not load Donut model: {e}")
        processor = model = None
        donut_available = False

    for path in images:
        name = os.path.basename(path)
        print(f"\n=== {name} ===")

        image = Image.open(path).convert("RGB")

        if donut_available:
            try:
                donut_result = run_donut(processor, model, image)
            except Exception as e:
                donut_result = f"[Donut inference failed: {e}]"
        else:
            donut_result = "[Donut unavailable]"
        print(f"Donut raw: {donut_result}")

        try:
            ocr_lines = run_ocr(image)
        except Exception as e:
            ocr_lines = [f"[OCR failed: {e}]"]
        print(f"OCR text lines: {ocr_lines}")


if __name__ == "__main__":
    main()
