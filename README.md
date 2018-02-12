# HoboCaptionNet
An encoder-decoder based deep neural network for image captioning

## Encoder
The encoder uses the Inception-v4 architecture to extract features from the image. The final softmax layer is removed and a fully-connected layer is added in its place which converts the extracted features to the same size as the word-embeddings and feeds it to the decoder network. The pre-trained inception-v4 weights are available [here](https://deepdetect.com/models/tf/inception_v4.pb). The weights must be places in the 'inception' directory.

## Decoder
The decoder network is a basic lstm network that takes in the output of the encoder and predits the most-probable word at each time-step with a maximum length of 20 words for any predicted caption.

## Pre-trained model
HoboCaptionNet was trained on a combination of COCO-2017 and flickr30k datasets using tensorflow-1.4. The pre-trained model is available [here](https://drive.google.com/file/d/1WJi66moaZuORmRjJhrAdKX7pes93dAKF/view?usp=sharing). The file must be placed in the 'model/Trained_Graphs' directory.

## Testing
To test the model pre-trained model, run the test.py script.
'''
python test.py --h
'''

Training and evaluation instructions will be updated soon.
