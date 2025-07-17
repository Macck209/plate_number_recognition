import json


def save_results(annots, detections, output_path='results.json'):
    results = []
    for filename, boxes in annots.items():
        true_text = boxes[0]['text'] if boxes else ''
        detected_text = detections.get(filename, '') # ignore empty detection ''
        if detected_text:
            max_len = max(len(true_text), len(detected_text))
            diff = ''
            for i in range(max_len):
                true_text_c = true_text[i] if i < len(true_text) else ''
                detected_text_c = detected_text[i] if i < len(detected_text) else ''
                diff += '_' if detected_text_c == true_text_c else detected_text_c or '_'

            result = {
                'detected': detected_text,
                'truth': true_text,
                'difference': diff,
                'filename': filename
            }
            results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f'Saved to: {output_path}')
