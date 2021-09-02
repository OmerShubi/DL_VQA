In this work we implement a Visual Question Answering (VQA) Model. 
The model utilizes a Convolution Network for Image feature Extraction, LSTM for Question feature extraction and finally makes use of Attention, before the final Fully Connected layers for prediction.

To recreate our model –
Run the `main.py` python script.

To recreate our results –
Run the `evaluate_vqa.py` python script.

Notes:
-	All the necessary paths and hyper parameters are configured in the `config.yaml` file for training the model and `config_eval.yaml` for the evaluating script. 
-	The model and logs are created automatically.
-	we assume the VQA 2.0 Dataset tobe in the path specified in the aforementioned config file.
-	We apply preprocessing steps to the text and images. The processed data is saved as an h5 file. If it does not exist the script will recreate the files, which may take a couple of hours.
-	Necessary packages include Schema, Hydra, Hydra-ax, pytorch.
