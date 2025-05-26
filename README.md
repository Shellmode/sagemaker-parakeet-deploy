# SageMaker Parakeet ASR Deployment

This repository contains code and instructions for deploying NVIDIA's Parakeet ASR (Automatic Speech Recognition) model on Amazon SageMaker.

## Overview

The project demonstrates how to deploy NVIDIA's Parakeet TDT 0.6B v2 speech recognition model as a SageMaker endpoint for real-time speech-to-text transcription. The deployment uses GPU-accelerated instances for optimal inference performance.

## Model Information

- **Model**: NVIDIA Parakeet TDT 0.6B v2
- **Source**: HuggingFace (`nvidia/parakeet-tdt-0.6b-v2`)
- **Type**: Automatic Speech Recognition (ASR)
- **Features**: Transcribes speech audio to text with high accuracy

## Repository Structure

```
.
├── README.md                           # This documentation file
├── sagemaker_parakeet_deploy.ipynb     # Main deployment notebook
├── code/                               # Inference code directory
│   ├── inference.py                    # SageMaker inference handler
│   └── requirements.txt                # Python dependencies
└── 2086-149220-0033.wav                # Sample audio file for testing
```

## Quick Start

The simplest way to deploy the model is to follow the steps in the `sagemaker_parakeet_deploy.ipynb` notebook:

1. Open the notebook in a SageMaker notebook instance or SageMaker Studio
2. Run all cells in sequence to:
   - Install dependencies
   - Package and upload the model code
   - Create and deploy the SageMaker endpoint
   - Test the endpoint with the sample audio file

## Deployment Process

The deployment process consists of these main steps:

1. **Environment Setup**: Install required packages and configure AWS credentials
2. **Code Packaging**: Package the inference code into a tarball and upload to S3
3. **Model Creation**: Create a SageMaker model using the PyTorch inference container
4. **Endpoint Configuration**: Configure the endpoint with appropriate instance type
5. **Endpoint Deployment**: Deploy the model to a SageMaker endpoint
6. **Testing**: Test the endpoint with sample audio

## Inference Code

The inference implementation in `code/inference.py` handles:

- Loading the Parakeet model from HuggingFace
- Processing binary audio input
- Running speech recognition
- Returning transcription results with timing information

## Requirements

- AWS account with SageMaker access
- IAM role with appropriate SageMaker permissions
- GPU instance availability in your AWS region (ml.g5.2xlarge recommended)

## Testing the Endpoint

The notebook includes code to test the endpoint with a sample audio file. The endpoint accepts binary audio data and returns a JSON response with:

- Transcription text
- Processing time in milliseconds

## Cleanup

To avoid incurring unnecessary charges, remember to delete the SageMaker resources when not in use:

```python
sm.delete_endpoint(EndpointName=endpoint_name)
sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm.delete_model(ModelName=model_name)
```

## Performance Considerations

- The model is deployed on GPU instances for optimal performance
- Inference time is typically under 200ms for short audio clips
- Longer audio files will require proportionally more processing time

## License

Please refer to the NVIDIA Parakeet model license for usage restrictions and terms.
