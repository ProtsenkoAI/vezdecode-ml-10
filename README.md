# Вездекод ML 10
## before you start
* install pipenv
* Some samples that I used for testing are provided at test_model_samples/ directory
* [link to pretrained model and config](https://drive.google.com/drive/folders/13x5stOckRVmycJ4kzQX9t8HZc3jIWOEZ?usp=sharing)
  (need to insert them to models/math-classifier/ after git clone)
## install & run
```
git clone git@github.com:ProtsenkoAI/vezdecode-ml-10.git
cd vezdecode-ml-10/
pipenv install
pipenv shell
mkdir -p models/math-classifier
```
At this moment insert model and config to models/math-classifier/
```
python boundingbox.py <PATH TO JPG TO INPUT> <PATH TO OUT DIRECTORY>
// ready-to-use example of command above:
python boundingbox.py ./test_model_samples/6.jpg ./predictions/
```

