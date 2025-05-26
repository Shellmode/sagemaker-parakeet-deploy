import nemo.collections.asr as nemo_asr
import logging
import json
import numpy as np
import traceback
import os
import time

# Configure logging to track execution and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the ASR model from HuggingFace
    """
    try:
        # Load pre-trained ASR model from Hugging Face
        model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        logger.info("Model loaded successfully: nvidia/parakeet-tdt-0.6b-v2")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def input_fn(request_body, request_content_type):
    """
    Process the incoming request data.
    Converts the raw binary audio data to a numpy array that can be processed by the model.
    """
    try:
        logger.info(f"Processing input with content type: {request_content_type}")
        # Validate content type - should be binary audio data
        if request_content_type != "application/octet-stream":
            logger.warning(f"Expected content type application/octet-stream but got {request_content_type}")

        # Convert binary data to numpy array (16-bit PCM audio format)
        audio_data = np.frombuffer(request_body, dtype=np.int16)
        logger.info(f"Input processed: audio data shape {audio_data.shape}, type {type(audio_data)}")
        return audio_data
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_fn(input_data, model):
    """
    Perform inference using the loaded model.
    Takes processed audio data and runs speech recognition to generate transcription.
    Includes performance timing to measure transcription latency.
    """
    try:
        logger.info("Starting prediction")
        audio_data = input_data
        logger.info(f"Audio data type: {type(audio_data)}, shape: {audio_data.shape}")

        # Measure transcription performance (millisecond precision)
        start_time = time.time()

        # Perform speech recognition
        output = model.transcribe([audio_data])

        # Calculate and log execution time
        end_time = time.time()
        transcribe_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.info(f"Transcribe execution time: {transcribe_time_ms:.2f} ms")

        logger.info(f"Prediction completed: {output}")
        # Return both the transcription text and the processing time
        return {"result": output[0].text, "transcribe_time_ms": transcribe_time_ms}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        # Return error information in case of failure
        return {"error": str(e), "result": ""}

def output_fn(prediction, response_content_type):
    """
    Format the prediction output according to the requested content type.
    Converts the prediction dictionary to the appropriate format for the response.
    """
    try:
        logger.info(f"Formatting output with content type: {response_content_type}")
        if response_content_type == "application/json":
            return json.dumps(prediction)
        else:
            # Default to JSON if content type is not supported
            logger.warning(f"Unsupported content type: {response_content_type}, defaulting to application/json")
            return json.dumps(prediction)
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        logger.error(traceback.format_exc())
        # Ensure we return something even if formatting fails
        return json.dumps({"error": "Error formatting output", "details": str(e)})
