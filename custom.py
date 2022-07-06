from pathlib import Path
import json
import time
import numpy_serializer as ns

# Step 1: Add your imports
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2FeatureExtractor, AutomaticSpeechRecognitionPipeline, Wav2Vec2CTCTokenizer


# Step 2: Fill these out for your model
model_name = 'wav2vec2-base-960h'


def load_model(code_dir):
    code_dir = Path(code_dir)
    model_dir = code_dir/model_name

    # Step 3.1 Load your tokenizer and model from model_dir
    model = AutoModelForCTC.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    tokenizer = Wav2Vec2CTCTokenizer(model_dir/'vocab.json')
    pipe = AutomaticSpeechRecognitionPipeline(model=model, processor=processor, feature_extractor=feature_extractor, tokenizer=tokenizer)
    return pipe


def score_unstructured(model, data, query, **kwargs):
    # Handle incoming data
    if not data:
        return data
    if isinstance(data, bytes):
        data = data.decode("utf8")

    # Get data from json
    data_dict = json.loads(data)
    buffer = ns.from_bytes(data_dict['buffer'].encode('latin-1'))
    args = data_dict['args']

    time_before_model = time.perf_counter()

    # This calls your function
    prediction = model(buffer, **args)

    time_after_model = time.perf_counter()

    # Structure json output
    output_dict = {
        'prediction': prediction,
        'model_run_time_seconds': time_after_model - time_before_model,
        }

    # Serialized output
    serialized_output = json.dumps(output_dict)
    return serialized_output



if __name__ == '__main__':
    proj_dir = Path(__file__).parent
    model = load_model(proj_dir)
    test_file = 'jsons/json_data_10_08_44.json'

    with open(proj_dir/test_file) as json_file:
        data = json.load(json_file)

    preds = score_unstructured(model=model, data=json.dumps(data), query=None)
    print(preds)