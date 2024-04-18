# Deploy your own Machine Learning API

For this project I created my own Machine learning model and deployed into its own API

- The web framework used to create the API was Fastapi 
- The model that was trained/tested/validated was Random forrest

## Problem Statement
  [Shanggong Medical Technology Co., Ltd.](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?resource=download) gathered data from various hospitals and medical centers across China to compile the Ocular Disease Intelligent Recognition (ODIR) dataset, which encompasses “real-life” patient information for 5,000 individuals. Each contributing institution captured images of the fundus, the back part of the eye opposite the lens, from each patient. Different cameras captured these images, resulting in various image resolutions. Trained professionals received the images and classified them into 8 different categories: normal, diabetes, glaucoma, cataracts, age-related macular degeneration, pathological myopia, other diseases/abnormalities. My goal for this project is to take the patient demographics of age and gender to predict at what age men and women are at risk of developing different eye diseases.

  The reason I picked a random forest model is that it’s great for using data that’s well-organized and doesn’t need much tweaking beforehand. There isn’t a straightforward link between the data points and eye conditions in our dataset. What’s great about this model is that it also points out which parts of the data are key in predicting the eye conditions, making it easier to figure out what’s important.
