# eegLLM
EEG classification and regression tasks integrating an LLM for explainability.

## Getting Started
* Clone repository
  ``` ssh
  https://github.com/ZaraCook/eegLLM.git
  ```
  
* Create  and activate conda environment
  ``` ssh
  conda env create -f eeg.yml
  ```
  ``` ssh
  conda activate eeg
  ```
* Install requirements
  ``` ssh
  pip install -r requirements.txt
  ```

## Preprocessing
* Change the following in preprocess.py
  ``` ssh
  fif_folder = "your_eeg_data_folder"
  csv_file = "your_eeg_label_csv_file"
  ```
  
* Run preprocess.py
  ``` ssh
  python preprocess.py
  ```

## Training the Model and Report Generation
* Request acces to [llama-3.2-3b-instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on Huggingface
* Once you have been granted access, generated an access token set it as an environment variable in your terminal
  ```ssh
  export HUGGINGFACE_TOKEN='your_hugging_face_token'
  ```
* Run eegConformer.py
  ```ssh
  python eegLLM.py
  ```
* The reports will be generated for 10 of the eeg scans in your sspecified report out directory (reports_dir)

