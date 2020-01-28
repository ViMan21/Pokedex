# Pokedex

This purpose of this project is to simulate a real-life PokeDex as seen in various Pokemon games and TV shows. When the first pokemon game released in 1996, the concept of a PokeDex was just as fictional as the Pokemon in the game. Thanks to advancement in Machine Learning and Computer Vision, we are now able to create an application that behaves similarily to the PokeDex.

## Data
The data to train the model was obtained by web scraping google images. A notebook was used to find the first 300 image links in Google Images for each of the first 151 pokemon. Next I transfered all the files containing the links to the Paperspace server where I created another notebook to download all the images in each file.
Next I cleaned the data by removing images that were not jpeg or png, and resizing them to all the same size. Google Images does not always produce images of what you searched, so I also manually inspected the data and removed any images that were mislabelled.

## Model
The model was trained using the ResNet50 weights. I tried using larger version of ResNet because I was trying to classify 151 image but that caused my GPU to run out of memory so I decided to settle with ResNet50. The data was trained for 8 epochs, then I unfroze the entire model and trained it for another 2 epochs. 

## Results
The model had an overall accuracy of 86%. Looking at were the model had the most common false prediction, it was for pokemon within the same evolution line. (eg. Pigeotto & Pigeot, Horsea & Seadra).This could be improved by cleaning the data further.

## Deploy
To run the application locally simply run the following command:
python app/server.py serve

This application was deployed on GCP using the Google App Engine. It can be accessed using the following link: http://pokemon-detector-pokedex.appspot.com/
