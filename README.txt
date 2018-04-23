Spotifeel - Spotify extension using Facial Recognition & Emotion 

Dependencies:
    keras==2.0.5
    tensorflow==1.1.0
    pandas==0.19.1
    numpy==1.12.1
    h5py==2.7.0
    statistics
    opencv2-python==3.2.0
    spotipy (command: 'easy_install spotipy' or 'pip install spotipy')

Instructions:
    1. Run the spotipy_classifier script after downloading all dependencies.
    2. Your default browser should open to Spotify.com where you will provide login credentials.
    3. For the sake of simplicity, we have hard-coded a temporary premium Spotify account that will be available until Saturday, May 12th 2018. 
        Username: 21knmlqzsjwosmjuvmhisfxaq
        Password: 6581041a
        (Note: For future use with your own Spotify Premium account, create a Spotify developer account
        and provide your own values for the variables surrounded by comments in 'spotipy_classifier')
    4. Provide the url in the browser from a successful login to the project console prompt.
    5. The webcam window will launch.
    6. Show a 'happy' emotion when you'd like to add your currently playing song
    to the Spotifeel playlist.
    7. Show an 'angry' reaction to skip the song you are currently playing.
    8. To further train emotion classification using the train_emotion_classifier script:
        Download the fer2013.tar.gz file from 
            https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
        Move the downloaded file to the datasets directory inside this repository.
        Untar the file:
            tar -xzf fer2013.tar
        Run the train_emotion_classification.py file
            python3 train_emotion_classifier.py